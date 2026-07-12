# main training script.  Run:  python train.py

from model import build_model
from dataset import FormulaDataset, collate_train
from vocab import Vocab, build_vocab
import config as C
import nltk
import matplotlib.pyplot as plt
import os
import shutil
import sys
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # no GUI needed

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ----- logging -----

DRIVE_DIR = "/content/drive/MyDrive/visual-latex-decompiler"


class Tee:
    # print to the console and the log file at the same time
    def __init__(self, path, stream):
        self.stream = stream
        self.file = open(path, "a")

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def save_to_drive(path, subdir):
    # same as the checkpoints, copy a file into the drive folder (colab only)
    if not os.path.exists("/content/drive"):
        return
    dst = os.path.join(DRIVE_DIR, subdir)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(path, dst)


def restore_log_from_drive(log_path):
    # pull the drive log back before appending, else the empty local log on a
    # fresh vm overwrites it
    src = os.path.join(DRIVE_DIR, "logs", os.path.basename(log_path))
    if os.path.exists(src) and not os.path.exists(log_path):
        shutil.copy2(src, log_path)
        print("restored log from drive:", src)


# ----- metric helpers -----

def calc_bleu(refs, hyps, max_n=4):
    smooth = nltk.translate.bleu_score.SmoothingFunction().method1
    total, cnt = 0.0, 0
    for r, h in zip(refs, hyps):
        n = min(max_n, len(r), len(h))
        if n == 0:
            continue
        w = [1.0 / n] * n
        try:
            s = nltk.translate.bleu_score.sentence_bleu(
                [r], h, weights=w, smoothing_function=smooth)
        except:
            s = 0.0
        total += s
        cnt += 1
    return total / cnt if cnt else 0.0


def calc_edit_dist(refs, hyps):
    """own levenshtein so we dont need the distance package at train time.
    single-row DP to save memory."""
    def _lev(a, b):
        na, nb = len(a), len(b)
        row = list(range(nb + 1))
        for i in range(1, na + 1):
            prev = row[0]
            row[0] = i
            for j in range(1, nb + 1):
                old = row[j]
                if a[i-1] == b[j-1]:
                    row[j] = prev
                else:
                    row[j] = 1 + min(prev, row[j], row[j-1])
                prev = old
        return row[nb]

    tot_d, tot_l = 0, 0
    for r, h in zip(refs, hyps):
        m = max(len(r), len(h))
        if m == 0:
            continue
        tot_d += _lev(r, h)
        tot_l += m
    return tot_d / tot_l if tot_l else 1.0


def get_tf_ratio(epoch):
    # linear decay from TF_START to TF_END
    t = epoch / max(1, C.EPOCHS - 1)
    return C.TF_START + t * (C.TF_END - C.TF_START)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----- validation pass -----

@torch.no_grad()
def validate(model, loader, loss_fn, vocab, dev):
    model.eval()
    tot_loss, n = 0.0, 0
    all_r, all_h = [], []
    decoded = 0

    for imgs, tgts, lens in loader:
        imgs, tgts = imgs.to(dev), tgts.to(dev)
        with torch.autocast("cuda", enabled=C.USE_AMP and dev.type == "cuda"):
            logits = model(imgs, tgts, tf_ratio=0.0)

            # skip SOS column for loss
            loss = loss_fn(
                logits[:, 1:].reshape(-1, logits.size(-1)),
                tgts[:, 1:].reshape(-1),
            )
        tot_loss += loss.item()
        n += 1

        # greedy decoding is the slow part (on colab i tried and each epoch has 40 min greedy check with best model and its awful!), so only do the first
        # VAL_DECODE_SAMPLES images -- loss already covers the full set. cap
        # the length near the longest target so a model that never emits EOS
        # can't spin for MAX_SEQ steps every batch.
        if decoded < C.VAL_DECODE_SAMPLES:
            cap = min(C.MAX_SEQ, int(lens.max().item()) + 20)
            with torch.autocast("cuda", enabled=C.USE_AMP and dev.type == "cuda"):
                preds = model.greedy(imgs, vocab.sos_id,
                                     vocab.eos_id, max_len=cap)
            for b in range(imgs.size(0)):
                ref_tok = vocab.decode(tgts[b].tolist())
                hyp_tok = vocab.decode(preds[b])
                all_r.append(ref_tok)
                all_h.append(hyp_tok)
            decoded += imgs.size(0)

    avg_loss = tot_loss / max(n, 1)
    bleu = calc_bleu(all_r, all_h, C.BLEU_N)
    edit = calc_edit_dist(all_r, all_h)
    return avg_loss, bleu, edit


# ----- one epoch of training -----

def train_epoch(model, loader, loss_fn, optim, scaler, epoch, dev, gstep):
    model.train()
    tot_loss, n = 0.0, 0
    tf = get_tf_ratio(epoch)
    skipped = 0
    amp_on = C.USE_AMP and dev.type == "cuda"

    for step, (imgs, tgts, lens) in enumerate(loader):
        imgs, tgts = imgs.to(dev), tgts.to(dev)
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp_on):
            logits = model(imgs, tgts, tf_ratio=tf)
            loss = loss_fn(
                logits[:, 1:].reshape(-1, logits.size(-1)),
                tgts[:, 1:].reshape(-1),
            )

        # if one batch goes nan and i let adam step on it, the m/v buffers get
        # wrecked and then every step after that is nan too -- thats what killed
        # my last run. so i just skip the bad batch and carry on.
        if not torch.isfinite(loss):
            optim.zero_grad(set_to_none=True)
            gstep += 1
            skipped += 1
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), C.CLIP)
        # clipping cant help if the grads are already nan (the norm is nan too),
        # so i check it here as well before stepping. under amp an occasional
        # inf is normal -- update() just shrinks the loss scale.
        if not torch.isfinite(gnorm):
            optim.zero_grad(set_to_none=True)
            scaler.update()
            gstep += 1
            skipped += 1
            continue

        # warmup -- scale the lr up over the first WARMUP_STEPS, then leave it
        # to the scheduler. the transformer hates getting the full lr on step 0.
        if gstep < C.WARMUP_STEPS:
            scale = (gstep + 1) / C.WARMUP_STEPS
            for g in optim.param_groups:
                g["lr"] = C.LR * scale

        scaler.step(optim)
        scaler.update()
        gstep += 1

        tot_loss += loss.item()
        n += 1
        if (step + 1) % C.LOG_INTERVAL == 0:
            print("  ep {} step {} loss={:.4f} tf={:.2f}".format(
                epoch+1, step+1, tot_loss/n, tf))

    if skipped:
        print("  ep {} skipped {} nan step(s)".format(epoch+1, skipped))
    return tot_loss / max(n, 1), gstep


# ----- plotting -----

def plot_curves(hist):
    os.makedirs(C.PLOTS_DIR, exist_ok=True)
    if not hist["tl"]:
        print("No training history to plot.")
        return
    eps = range(1, len(hist["tl"]) + 1)

    # per-method filenames so rnn and transformer runs don't clobber each
    # other -- loss_rnn.png vs loss_transformer.png (same convention as the
    # checkpoints and prediction files)
    suffix = C.MODEL_TYPE

    # loss
    fig, ax = plt.subplots()
    ax.plot(eps, hist["tl"], label="train")
    ax.plot(eps, hist["vl"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss ({})".format(suffix))
    ax.legend()
    fig.savefig(os.path.join(
        C.PLOTS_DIR, "loss_{}.png".format(suffix)), dpi=140)
    plt.close(fig)

    # bleu
    fig, ax = plt.subplots()
    ax.plot(eps, hist["vb"], "g")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU")
    ax.set_title("Validation BLEU ({})".format(suffix))
    fig.savefig(os.path.join(
        C.PLOTS_DIR, "bleu_{}.png".format(suffix)), dpi=140)
    plt.close(fig)

    # edit dist
    fig, ax = plt.subplots()
    ax.plot(eps, hist["ve"], "r")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Edit Dist")
    ax.set_title("Validation Edit Distance ({})".format(suffix))
    fig.savefig(os.path.join(
        C.PLOTS_DIR, "edit_distance_{}.png".format(suffix)), dpi=140)
    plt.close(fig)

    print("Plots saved to", C.PLOTS_DIR)


# ----- main -----

def main():
    # send every print into logs/ as well so the whole run is saved
    os.makedirs(C.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(
        C.LOGS_DIR, "train_log_{}.txt".format(C.MODEL_TYPE))
    restore_log_from_drive(log_path)
    real_stdout = sys.stdout
    sys.stdout = Tee(log_path, real_stdout)
    try:
        run(log_path)
    finally:
        # flush + push the log even if i stop the run early
        sys.stdout.flush()
        save_to_drive(log_path, "logs")
        sys.stdout = real_stdout


def run(log_path):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    # vocab
    if os.path.exists(C.VOCAB_PATH):
        vocab = Vocab.load()
    else:
        vocab, _ = build_vocab()
        vocab.save()

    # data
    train_ds = FormulaDataset(
        C.TRAIN_IMAGES_DIR, C.TRAIN_FORMULAS, vocab, training=True)
    val_ds = FormulaDataset(
        C.VAL_IMAGES_DIR,   C.VAL_FORMULAS,   vocab, training=False)

    # keep workers alive between epochs so we don't re-fork them every time
    train_ld = DataLoader(train_ds, C.BATCH, shuffle=True,
                          num_workers=C.WORKERS, pin_memory=C.PIN_MEM,
                          persistent_workers=C.WORKERS > 0, collate_fn=collate_train)
    val_ld = DataLoader(val_ds,   C.BATCH, shuffle=False,
                        num_workers=C.WORKERS, pin_memory=C.PIN_MEM,
                        persistent_workers=C.WORKERS > 0, collate_fn=collate_train)

    # model  (rnn or transformer, decided by C.MODEL_TYPE)
    model = build_model(len(vocab)).to(dev)
    print("Model type:", C.MODEL_TYPE)
    print("Parameters:", "{:,}".format(n_params(model)))
    if C.MODEL_TYPE == "rnn":
        print("Attention:", C.USE_ATTN)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_id, label_smoothing=C.LABEL_SMOOTH)
    optim = torch.optim.Adam(model.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.StepLR(optim, C.LR_STEP, C.LR_GAMMA)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=C.USE_AMP and dev.type == "cuda")

    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)

    hist = {"tl": [], "vl": [], "vb": [], "ve": []}
    best_vloss = float("inf")
    start_ep = 0

    # resume from the "last" checkpoint (full state, saved every epoch) so a
    # stopped run continues from the true latest epoch. "best" is weights-only
    # for predict.py. names carry the method so rnn/transformer don't clobber.
    last_path = os.path.join(
        C.CHECKPOINT_DIR, "model_last_{}.pt".format(C.MODEL_TYPE))
    best_path = os.path.join(
        C.CHECKPOINT_DIR, "model_best_{}.pt".format(C.MODEL_TYPE))
    resume_path = last_path if os.path.exists(last_path) else best_path
    print("Looking for checkpoint:", resume_path,
          "exists:", os.path.exists(resume_path))
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=dev)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            raise SystemExit(
                "Checkpoint {} does not match the current architecture "
                "(config changed, e.g. CNN_POOLS). Delete the old "
                "checkpoints (local AND the copies on Drive) to train from "
                "scratch.\n\nOriginal error: {}".format(resume_path, e))
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(ckpt["epoch"]):
                    sched.step()
        start_ep = ckpt["epoch"]
        best_vloss = ckpt.get("best_vl", ckpt["vl"])
        if "hist" in ckpt:
            hist = ckpt["hist"]
        if start_ep >= C.EPOCHS:
            print("Already trained {}/{} epochs. Increase EPOCHS in config.py to continue.".format(
                start_ep, C.EPOCHS))
        print("Resumed from epoch {}, best val_loss={:.4f}".format(
            start_ep, best_vloss))

    # global optimizer-step counter drives the lr warmup. seed it from the
    # resume point so a run continued past warmup doesn't warm up again.
    gstep = start_ep * len(train_ld)

    for ep in range(start_ep, C.EPOCHS):
        t0 = time.time()
        tl, gstep = train_epoch(
            model, train_ld, loss_fn, optim, scaler, ep, dev, gstep)
        vl, vb, ve = validate(model, val_ld, loss_fn, vocab, dev)
        sched.step()
        dt = time.time() - t0

        print("Epoch {}/{} tl={:.4f} vl={:.4f} BLEU={:.4f} ED={:.4f} lr={:.6f} {:.0f}s".format(
            ep+1, C.EPOCHS, tl, vl, vb, ve, optim.param_groups[0]["lr"], dt))

        hist["tl"].append(tl)
        hist["vl"].append(vl)
        hist["vb"].append(vb)
        hist["ve"].append(ve)

        # save the log to drive after every epoch so i can download it later
        sys.stdout.flush()
        save_to_drive(log_path, "logs")

        is_best = vl < best_vloss
        if is_best:
            best_vloss = vl

        # full training state, written every epoch
        torch.save({
            "epoch": ep + 1,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "hist": hist,
            "vl": vl, "vb": vb, "ve": ve,
            "best_vl": best_vloss,
            "vocab_size": len(vocab),
        }, last_path)
        print("  saved:", last_path)
        save_to_drive(last_path, "checkpoints")

        if is_best:
            # weights only -- smaller file, quicker drive sync
            torch.save({
                "epoch": ep + 1,
                "model": model.state_dict(),
                "vl": vl, "vb": vb, "ve": ve,
                "vocab_size": len(vocab),
            }, best_path)
            print("  saved:", best_path)
            save_to_drive(best_path, "checkpoints")

        if os.path.exists("/content/drive"):
            print("  synced to Drive!")

    plot_curves(hist)
    print("\n--- Done ---")
    print("Best val loss: {:.4f}".format(best_vloss))
    if hist["vb"]:
        print("Final BLEU:    {:.4f}".format(hist["vb"][-1]))
        print("Final ED:      {:.4f}".format(hist["ve"][-1]))
    else:
        print("(No new epochs ran — metrics from checkpoint only)")


if __name__ == "__main__":
    main()
