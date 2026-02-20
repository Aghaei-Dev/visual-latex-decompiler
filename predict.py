# generate latex predictions for test images.
#
# usage:
#   python predict.py                          # greedy
#   python predict.py --beam                   # beam search
#   python predict.py --beam --postprocess     # beam + cleanup

import os
import argparse
import torch
from torch.utils.data import DataLoader

import config as C
from vocab import Vocab
from dataset import TestDataset, collate_test
from model import Im2Latex
from postprocess import clean_latex


def args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", default=os.path.join(C.CHECKPOINT_DIR, "model_best.pt"))
    p.add_argument("--beam", action="store_true", help="use beam search")
    p.add_argument("--beam-width", type=int, default=C.BEAM_K)
    p.add_argument("--output", default=C.OUTPUT_FORMULAS)
    p.add_argument("--postprocess", action="store_true", help="clean up LaTeX")
    return p.parse_args()


def main():
    opt = args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    vocab = Vocab.load()

    ckpt = torch.load(opt.checkpoint, map_location=dev)
    model = Im2Latex(ckpt["vocab_size"]).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Loaded {} (epoch {})".format(opt.checkpoint, ckpt["epoch"]))

    ds = TestDataset(C.TEST_IMAGES_DIR)
    loader = DataLoader(ds, C.BATCH, shuffle=False,
                        num_workers=C.WORKERS, collate_fn=collate_test)

    predictions = []

    done = 0
    total = len(ds)
    for imgs, fnames in loader:
        done += imgs.size(0)
        print("\r  {}/{} images ({:.0f}%)".format(done,
              total, 100*done/total), end="", flush=True)
        imgs = imgs.to(dev)
        if opt.beam:
            seqs = model.beam_decode(
                imgs, vocab.sos_id, vocab.eos_id, beam_k=opt.beam_width)
        else:
            seqs = model.greedy(imgs, vocab.sos_id, vocab.eos_id)

        for i, seq in enumerate(seqs):
            toks = vocab.decode(seq)
            formula = " ".join(toks)
            if opt.postprocess:
                formula = clean_latex(formula)
            base = os.path.splitext(fnames[i])[0]
            try:
                key = int(base)
            except:
                key = base
            predictions.append((key, formula))

    predictions.sort(key=lambda x: x[0])

    with open(opt.output, "w", encoding="utf-8") as f:
        for _, formula in predictions:
            f.write(formula + "\n")

    print("Wrote {} predictions -> {}".format(len(predictions), opt.output))


if __name__ == "__main__":
    main()
