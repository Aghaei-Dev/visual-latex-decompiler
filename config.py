# all the paths and hyper-params lives here.
# whenever i want to change something i just come here easy and nice :)
# instead of hunting through every file.

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
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# --- image size (you told us in the project details) ---
IMG_H = 64
IMG_W = 256
IMG_CHANNELS = 1  # grayscale

# --- special tokens for vocab ---
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

# --- which method are we running ---
# "rnn"          -> method 1: biLSTM encoder + lstm decoder with attention - Computer Vision Project (Dr.Adeleh Bitarafan) 4041
# "transformer"  -> method 2: transformer encoder + transformer decoder, same cnn front-end - Deep Learning Project (Dr.Kazem Fouladi) 4042
# just change this and re-run train.py / predict.py, the rest follows.
MODEL_TYPE = "transformer"

# --- CNN ---
# four conv blocks, doubles channels each time  (shared by both methods)
CNN_FILTERS = [64, 128, 256, 512]
# pool (h, w) per block -- only halve width twice (256 -> 64 columns) so the
# decoder has enough feature columns to attend over
CNN_POOLS = [(2, 2), (2, 2), (2, 1), (2, 1)]

# --- row encoder (biLSTM on top of CNN columns) ---
ENC_HIDDEN = 256
ENC_LAYERS = 1
ENC_DROP = 0.1

# --- decoder LSTM ---
EMBED_DIM = 128
DEC_HIDDEN = 512
DEC_LAYERS = 1
DEC_DROP = 0.2
MAX_SEQ = 200   # longest formula the decoder will generate

# attention toggle on/off  (set False for the no-attention baseline)
USE_ATTN = True
ATTN_DIM = 256

# --- transformer (method 2) ---
# same cnn front-end, but the whole seq2seq is done with attention:
# a transformer encoder over the feature columns + a transformer decoder.
# i keep d_model the same width as the biLSTM encoder (ENC_HIDDEN*2 = 512)
# so the two methods stay roughly the same size and the comparison is fair.
TRANS_D_MODEL = 512
TRANS_HEADS = 8       # 512 / 8 = 64 dims per head
TRANS_FF = 2048       # feed-forward width inside each block
# 6 layers -- the 4+4 run plateaued at BLEU ~0.90 with zero overfit gap
TRANS_ENC_LAYERS = 6  # column self-attention blocks (replaces the biLSTM)
TRANS_DEC_LAYERS = 6  # decoder blocks (self-attention + cross-attention)
TRANS_DROP = 0.1

# --- training ---
BATCH = 192      # 3*64 so still tensor-core friendly :) always they told us it must be power of 2 
EPOCHS = 30
LR = 3e-4 if MODEL_TYPE == "transformer" else 1e-3   # transformer needs it lower
LR_STEP = 10     # drop lr every N epochs
LR_GAMMA = 0.5
CLIP = 5.0       # gradient clip
LABEL_SMOOTH = 0.1

USE_AMP = True   # mixed precision -- ~2x faster per step on the gpu

# ramp the lr up over the first steps -- full lr from step 0 blows up to nan
WARMUP_STEPS = 500

# teacher forcing -- starts high, decays linearly to TF_END (rnn only,
# the transformer is always fully teacher forced)
TF_START = 1.0
TF_END = 0.6

WORKERS = 2
PIN_MEM = True

# --- eval stuff ---
BLEU_N = 4
LOG_INTERVAL = 50
# only decode this many val images for BLEU/edit-dist -- loss uses the full set
VAL_DECODE_SAMPLES = 640

# --- beam search ---
BEAM_K = 5
BEAM_LEN_NORM = 0.7  # score / len^alpha, else short beams always win
