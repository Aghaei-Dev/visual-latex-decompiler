# vocab.py
# Handles building the token vocabulary from training formulas.
# Also saves/loads from disk so we only build it once.

import pickle
import config as C


class Vocab:
    """Maps LaTeX tokens <-> integer indices."""

    def __init__(self):
        self.tok2id = {}
        self.id2tok = {}
        self.size = 0
        # add special tokens right away (order matters: PAD must be 0)
        for s in [C.PAD, C.SOS, C.EOS, C.UNK]:
            self._insert(s)

    def _insert(self, token):
        if token in self.tok2id:
            return
        i = self.size
        self.tok2id[token] = i
        self.id2tok[i] = token
        self.size += 1

    # convenience
    @property
    def pad_id(self): return self.tok2id[C.PAD]
    @property
    def sos_id(self): return self.tok2id[C.SOS]
    @property
    def eos_id(self): return self.tok2id[C.EOS]
    @property
    def unk_id(self): return self.tok2id[C.UNK]
    def __len__(self): return self.size

    def encode(self, tokens):
        """list of strings -> list of ints"""
        return [self.tok2id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids, skip_special=True):
        """list of ints -> list of strings"""
        out = []
        specials = {C.PAD, C.SOS, C.EOS}
        for i in ids:
            tok = self.id2tok.get(i, C.UNK)
            if skip_special and tok in specials:
                continue
            out.append(tok)
        return out

    def save(self, path=None):
        path = path or C.VOCAB_PATH
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print("Saved vocab ({} tokens) -> {}".format(self.size, path))

    @staticmethod
    def load(path=None):
        path = path or C.VOCAB_PATH
        with open(path, "rb") as f:
            v = pickle.load(f)
        print("Loaded vocab ({} tokens) <- {}".format(v.size, path))
        return v


# ---- reading the formulas txt ----

def read_formulas(fpath):
    """
    One formula per line. Returns list of lists-of-tokens.
    """
    formulas = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                formulas.append(stripped.split())
    return formulas


def build_vocab(path=None):
    """Read train formulas, count tokens, build Vocab object."""
    path = path or C.TRAIN_FORMULAS
    formulas = read_formulas(path)
    print("Found {} formulas in {}".format(len(formulas), path))

    # count frequencies
    freq = {}
    for f in formulas:
        for t in f:
            freq[t] = freq.get(t, 0) + 1

    v = Vocab()
    for tok in sorted(freq.keys()):
        v._insert(tok)

    print("Vocab: {} tokens total".format(v.size))
    return v, formulas


if __name__ == "__main__":
    v, fms = build_vocab()
    v.save()
    # quick test
    if fms:
        sample = fms[0]
        enc = v.encode(sample)
        dec = v.decode(enc)
        print("Original :", " ".join(sample[:15]))
        print("Encoded  :", enc[:15])
        print("Decoded  :", " ".join(dec[:15]))
