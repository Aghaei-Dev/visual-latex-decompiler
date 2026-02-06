# config.py
# =========
# I keep all the paths and hyperparams in one place
# so I don't have to dig through multiple files every time
# I want to change something.

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data_Im2Latx")

# --- data paths ---
TRAIN_FORMULAS = os.path.join(DATA_DIR, "train_formulas.txt")
VAL_FORMULAS = os.path.join(DATA_DIR, "validation_formulas.txt")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "images_train")
VAL_IMAGES_DIR = os.path.join(DATA_DIR, "images_val")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "images_test")

OUTPUT_FORMULAS = os.path.join(BASE_DIR, "test_formulas.txt")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.pkl")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# --- image size (given in the assignment) ---
IMG_H = 64
IMG_W = 256
IMG_CHANNELS = 1  # grayscale

# --- special tokens for the vocabulary ---
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

# --- CNN ---
# four conv blocks, each doubles the channel count
CNN_FILTERS = [64, 128, 256, 512]

# --- row encoder (biLSTM over the CNN feature columns) ---
ENC_HIDDEN = 256
ENC_LAYERS = 1
ENC_DROP = 0.1

# --- decoder ---
EMBED_DIM = 128
DEC_HIDDEN = 512
DEC_LAYERS = 1
DEC_DROP = 0.2
MAX_SEQ = 200   # longest sequence the decoder will ever produce

# attention (set to False for the base model without attention)
USE_ATTN = True
ATTN_DIM = 256

# --- training ---
BATCH = 32
EPOCHS = 40
LR = 1e-3
LR_STEP = 10     # drop every N epochs
LR_GAMMA = 0.5
CLIP = 5.0    # gradient clipping

TF_START = 1.0    # teacher-forcing ratio at start
TF_END = 0.6    # ... at end

WORKERS = 2      # dataloader workers
PIN_MEM = True

# --- eval ---
BLEU_N = 4
LOG_INTERVAL = 50
SAVE_BEST = True

# --- beam search ---
BEAM_K = 5
