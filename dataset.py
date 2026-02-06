# dataset.py
# PyTorch datasets for loading image-formula pairs and test images.

import os
import glob

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms

import config as C
from vocab import Vocab, read_formulas


def make_transform(training=False):
    """Build image preprocessing pipeline."""
    t = [
        transforms.Grayscale(1),
        transforms.Resize((C.IMG_H, C.IMG_W)),
    ]
    if training:
        # small augmentation so the model doesn't overfit too fast
        t.append(transforms.RandomAffine(degrees=1, translate=(0.02, 0.02)))
    t.append(transforms.ToTensor())          # -> [0,1]
    t.append(transforms.Normalize([0.5], [0.5]))  # -> [-1,1]
    return transforms.Compose(t)


class FormulaDataset(Dataset):
    """
    Training / validation set.
    Each image filename is just {index}.png where index corresponds
    to the line number (0-based) in the formulas txt file.
    """

    def __init__(self, img_dir, formulas_path, vocab, training=False):
        self.vocab = vocab
        self.tfm = make_transform(training)

        self.formulas = read_formulas(formulas_path)

        # match images to their formulas
        files_on_disk = set(os.listdir(img_dir))
        self.pairs = []  # (full_path, formula_index)
        for idx in range(len(self.formulas)):
            for ext in ("png", "jpg", "jpeg", "bmp"):
                name = "{}.{}".format(idx, ext)
                if name in files_on_disk:
                    self.pairs.append((os.path.join(img_dir, name), idx))
                    break
        print("[FormulaDataset] {} samples from {}".format(
            len(self.pairs), img_dir))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, fidx = self.pairs[i]
        img = Image.open(path).convert("L")
        img = self.tfm(img)

        toks = self.formulas[fidx]
        ids = [self.vocab.sos_id] + \
            self.vocab.encode(toks) + [self.vocab.eos_id]
        return img, torch.tensor(ids, dtype=torch.long)


class TestDataset(Dataset):
    """Test images only -- no labels."""

    def __init__(self, img_dir):
        self.tfm = make_transform(training=False)
        everything = glob.glob(os.path.join(img_dir, "*"))
        ok_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        everything = [p for p in everything if os.path.splitext(p)[
            1].lower() in ok_ext]

        # sort numerically by filename
        def num_key(p):
            n = os.path.splitext(os.path.basename(p))[0]
            try:
                return (0, int(n))
            except:
                return (1, n)
        self.paths = sorted(everything, key=num_key)
        print("[TestDataset] {} images from {}".format(
            len(self.paths), img_dir))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("L")
        return self.tfm(img), os.path.basename(p)


# --- collate fns for DataLoader ---

def collate_train(batch):
    imgs, tgts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    lens = torch.tensor([len(t) for t in tgts])
    tgts = pad_sequence(tgts, batch_first=True, padding_value=0)  # 0 = PAD
    return imgs, tgts, lens


def collate_test(batch):
    imgs, names = zip(*batch)
    return torch.stack(imgs, 0), names
