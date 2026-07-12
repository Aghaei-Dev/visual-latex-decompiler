# pytorch datasets for loading the image-formula pairs.

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
    t = [
        transforms.Grayscale(1),
        transforms.Resize((C.IMG_H, C.IMG_W)),
    ]
    if training:
        # very light augmentation so the model doesn't over-fits too fast -- cant go higher or the formulas get distorted
        # fill=255 -- shifted-in edges stay white, not ink-black
        t.append(transforms.RandomAffine(
            degrees=1, translate=(0.02, 0.02), fill=255))
    t.append(transforms.ToTensor())          # -> [0,1]
    t.append(transforms.Normalize([0.5], [0.5]))  # -> [-1,1]
    return transforms.Compose(t)


class FormulaDataset(Dataset):
    """train/val set. image filename = formula index (e.g. 00042.png -> line 42)."""

    def __init__(self, img_dir, formulas_path, vocab, training=False):
        self.vocab = vocab
        self.tfm = make_transform(training)
        self.formulas = read_formulas(formulas_path)

        # match images on disk to their formula index
        files_on_disk = set(os.listdir(img_dir))
        self.pairs = []
        for idx in range(len(self.formulas)):
            for ext in ("png", "jpg", "jpeg", "bmp"):
                name = "{:05d}.{}".format(idx, ext)
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
    """test images only -- no labels."""

    def __init__(self, img_dir):
        self.tfm = make_transform(training=False)
        everything = glob.glob(os.path.join(img_dir, "*"))
        ok_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        everything = [p for p in everything if os.path.splitext(p)[
            1].lower() in ok_ext]

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


def collate_train(batch):
    # pad target sequences to same length in the batch
    imgs, tgts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    lens = torch.tensor([len(t) for t in tgts])
    tgts = pad_sequence(tgts, batch_first=True, padding_value=0)  # 0 = PAD
    return imgs, tgts, lens


def collate_test(batch):
    imgs, names = zip(*batch)
    return torch.stack(imgs, 0), names
