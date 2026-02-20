# main training script.  Run:  python train.py

from model import Im2Latex
from dataset import FormulaDataset, collate_train
from vocab import Vocab, build_vocab
import config as C
import nltk
import matplotlib.pyplot as plt
import os
import shutil
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


# ----- metric helpers -----

def calc_bleu(refs, hyps, max_n=4):
    total, cnt = 0.0, 0
    for r, h in zip(refs, hyps):
        n = min(max_n, len(r), len(h))
        if n == 0:
            continue
        w = [1.0 / n] * n
        try:
            s = nltk.translate.bleu_score.sentence_bleu([r], h, weights=w)
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

    for imgs, tgts, lens in loader:
        imgs, tgts = imgs.to(dev), tgts.to(dev)
        logits = model(imgs, tgts, tf_ratio=0.0)

        # skip SOS column for loss
        loss = loss_fn(
            logits[:, 1:].reshape(-1, logits.size(-1)),
            tgts[:, 1:].reshape(-1),
        )
        tot_loss += loss.item()
        n += 1

        preds = model.greedy(imgs, vocab.sos_id, vocab.eos_id)
        for b in range(imgs.size(0)):
            ref_tok = vocab.decode(tgts[b].tolist())
            hyp_tok = vocab.decode(preds[b])
            all_r.append(ref_tok)
            all_h.append(hyp_tok)

    avg_loss = tot_loss / max(n, 1)
    bleu = calc_bleu(all_r, all_h, C.BLEU_N)
    edit = calc_edit_dist(all_r, all_h)
    return avg_loss, bleu, edit


# ----- one epoch of training -----

def train_epoch(model, loader, loss_fn, optim, epoch, dev):
    model.train()
    tot_loss, n = 0.0, 0
    tf = get_tf_ratio(epoch)

    for step, (imgs, tgts, lens) in enumerate(loader):
        imgs, tgts = imgs.to(dev), tgts.to(dev)
        optim.zero_grad()
        logits = model(imgs, tgts, tf_ratio=tf)
        loss = loss_fn(
            logits[:, 1:].reshape(-1, logits.size(-1)),
            tgts[:, 1:].reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), C.CLIP)
        optim.step()

        tot_loss += loss.item()
        n += 1
        if (step + 1) % C.LOG_INTERVAL == 0:
            print("  ep {} step {} loss={:.4f} tf={:.2f}".format(
                epoch+1, step+1, tot_loss/n, tf))

    return tot_loss / max(n, 1)


# ----- plotting -----

def plot_curves(hist):
    os.makedirs(C.PLOTS_DIR, exist_ok=True)
    if not hist["tl"]:
        print("No training history to plot.")
        return
    eps = range(1, len(hist["tl"]) + 1)

    # loss
    fig, ax = plt.subplots()
    ax.plot(eps, hist["tl"], label="train")
    ax.plot(eps, hist["vl"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    fig.savefig(os.path.join(C.PLOTS_DIR, "loss.png"), dpi=140)
    plt.close(fig)

    # bleu
    fig, ax = plt.subplots()
    ax.plot(eps, hist["vb"], "g")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU")
    ax.set_title("Validation BLEU")
    fig.savefig(os.path.join(C.PLOTS_DIR, "bleu.png"), dpi=140)
    plt.close(fig)

    # edit dist
    fig, ax = plt.subplots()
    ax.plot(eps, hist["ve"], "r")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Edit Dist")
    ax.set_title("Validation Edit Distance")
    fig.savefig(os.path.join(C.PLOTS_DIR, "edit_distance.png"), dpi=140)
    plt.close(fig)

    print("Plots saved to", C.PLOTS_DIR)


# ----- main -----

def main():
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

    train_ld = DataLoader(train_ds, C.BATCH, shuffle=True,
                          num_workers=C.WORKERS, pin_memory=C.PIN_MEM, collate_fn=collate_train)
    val_ld = DataLoader(val_ds,   C.BATCH, shuffle=False,
                        num_workers=C.WORKERS, pin_memory=C.PIN_MEM, collate_fn=collate_train)

    # model
    model = Im2Latex(len(vocab)).to(dev)
    print("Parameters:", "{:,}".format(n_params(model)))
    print("Attention:", C.USE_ATTN)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optim = torch.optim.Adam(model.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.StepLR(optim, C.LR_STEP, C.LR_GAMMA)

    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)

    hist = {"tl": [], "vl": [], "vb": [], "ve": []}
    best_vloss = float("inf")
    start_ep = 0

    # resume from checkpoint if available
    ckpt_path = os.path.join(C.CHECKPOINT_DIR, "model_best.pt")
    print("Looking for checkpoint:", ckpt_path,
          "exists:", os.path.exists(ckpt_path))
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(ckpt["epoch"]):
                    sched.step()
        start_ep = ckpt["epoch"]
        best_vloss = ckpt["vl"]
        if "hist" in ckpt:
            hist = ckpt["hist"]
        if start_ep >= C.EPOCHS:
            print("Already trained {}/{} epochs. Increase EPOCHS in config.py to continue.".format(
                start_ep, C.EPOCHS))
        print("Resumed from epoch {}, val_loss={:.4f}".format(start_ep, best_vloss))

    for ep in range(start_ep, C.EPOCHS):
        t0 = time.time()
        tl = train_epoch(model, train_ld, loss_fn, optim, ep, dev)
        vl, vb, ve = validate(model, val_ld, loss_fn, vocab, dev)
        sched.step()
        dt = time.time() - t0

        print("Epoch {}/{} tl={:.4f} vl={:.4f} BLEU={:.4f} ED={:.4f} lr={:.6f} {:.0f}s".format(
            ep+1, C.EPOCHS, tl, vl, vb, ve, optim.param_groups[0]["lr"], dt))

        hist["tl"].append(tl)
        hist["vl"].append(vl)
        hist["vb"].append(vb)
        hist["ve"].append(ve)

        is_best = vl < best_vloss
        if is_best:
            best_vloss = vl
        if is_best or not C.SAVE_BEST:
            tag = "best" if is_best else "ep{}".format(ep+1)
            path = os.path.join(C.CHECKPOINT_DIR, "model_{}.pt".format(tag))
            torch.save({
                "epoch": ep + 1,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "hist": hist,
                "vl": vl, "vb": vb, "ve": ve,
                "vocab_size": len(vocab),
            }, path)
            print("  saved:", path)

            # Sync to Google Drive immediately (Colab only)
            drive_ckpt = '/content/drive/MyDrive/CV_220704007/checkpoints'
            if os.path.exists('/content/drive'):
                os.makedirs(drive_ckpt, exist_ok=True)
                shutil.copy2(path, drive_ckpt)
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
