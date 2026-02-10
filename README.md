# Image to LaTeX -- Project Report -- 220704007

## 1. Problem

We need to convert a 64x256 grayscale image of a math formula into its LaTeX source code. This is an image-to-sequence problem: the input is a fixed-size image `64x256` but the output is a variable-length token sequence (:

### before preceding check the file structure:

![alt text](./images/ROUTING.png)

## 2. Dataset

All data lives in the `Data_Im2Latx/` folder:

| Split | Images          | Labels                                                   |
| ----- | --------------- | -------------------------------------------------------- |
| Train | `images_train/` | `train_formulas.txt`                                     |
| Val   | `images_val/`   | `validation_formulas.txt`                                |
| Test  | `images_test/`  | `I generated these it lives in the ./test_formulas.txt)` |

Image filenames are just the formula index (00000.png, 00001.png, ...). Formulas in the text files are one per line, tokens separated by spaces.
in simpler way the line number of .txt are the filename of each image.

### Preprocessing

- Grayscale, resized to 64x256
- Normalized to [-1, 1] (ToTensor gives [0,1], then Normalize([0.5],[0.5]) maps to [-1,1])
- Light augmentation during training: tiny rotation (1 degree) + shift (2%)
- Tokenized by whitespace; vocabulary built from training set
- Special tokens: `<PAD>` (0), `<SOS>` (1), `<EOS>` (2), `<UNK>` (3)

## 3. Neural Network Architecture

Based on Deng et al. (2016), "What You Get Is What You See." The model has three main parts connected in a pipeline:

```bash
Image (1x64x256) --> [CNN Encoder] --> [Row Encoder (biLSTM)] --> [Decoder (LSTM + Attention)] --> LaTeX tokens
```

### 3.1 CNN Encoder (`ConvEncoder` in model.py)

**What it does:** Extracts visual features from the input image. Each convolutional block learns to detect patterns -- edges in early layers, more complex shapes (symbols, strokes) in later layers.

**Structure:** Four blocks, each containing:

- `Conv2d(3x3, padding=1)` -- extracts local spatial features
- `BatchNorm2d` -- stabilizes training by normalizing activations
- `ReLU` -- non-linearity so the network can learn complex patterns
- `MaxPool2d(2x2)` -- reduces spatial size by half, keeps strongest features

**Channels progression:** 1 --> 64 --> 128 --> 256 --> 512

**Spatial size after each pool:**

| Layer  | Size     |
| ------ | -------- |
| Input  | 64 x 256 |
| Pool 1 | 32 x 128 |
| Pool 2 | 16 x 64  |
| Pool 3 | 8 x 32   |
| Pool 4 | 4 x 16   |

**Output:** Feature map of shape `(B, 512, 4, 16)` -- 512 channels, 4 rows, 16 columns.

### 3.2 Row Encoder (`RowEncoder` in `model.py`)

**What it does:** Reads the CNN feature map left-to-right (and right-to-left) to capture the sequential structure of the formula. A formula like `a + b = c` has a natural left-to-right order that the CNN alone cannot model.

**How it works:**

1. Each of the 16 columns is flattened: 512 channels x 4 rows = 2048-dim vector
2. These 16 vectors are fed as a sequence into a bidirectional LSTM
3. Forward LSTM reads left-to-right, backward LSTM reads right-to-left
4. Their outputs are concatenated

**Output:** Sequence of 16 vectors, each 512-dim (256 forward + 256 backward).

### 3.3 Decoder (`Decoder` in `model.py`)

**What it does:** Generates LaTeX tokens one at a time, starting from `<SOS>` and stopping at `<EOS>`.

**Each step:**

1. Embed the previous token (128-dim embedding)
2. Compute attention over encoder outputs --> context vector (what part of the image to look at)
3. Concatenate embedding + context, feed into LSTM (512 hidden units)
4. Concatenate LSTM output + context, project to vocabulary logits
5. Pick the token with highest probability

**Initial hidden state:** Learned projection of the mean encoder output (not zeros -- this gives the decoder a summary of the whole image to start).

### 3.4 Attention Mechanism (`Attention` in `model.py`)

**What it does:** Lets the decoder "look at" different parts of the image at each time step. Without attention, the decoder must compress the entire image into one hidden vector. With attention, it can focus on the relevant column (e.g., look at the left side when generating `\frac`, look at the numerator when generating the next symbol).

**Type:** Bahdanau additive attention.

**Formula:**

```bash
score_i = v^T * tanh(W_enc * h_enc_i + W_dec * h_dec)
weights = softmax(scores)           -- how much to attend to each position
context = sum(weights_i * h_enc_i)  -- weighted combination of encoder outputs
```

**Parameters:**

- `W_enc`: Linear(512 --> 256) -- projects encoder states
- `W_dec`: Linear(512 --> 256) -- projects decoder hidden state
- `v`: Linear(256 --> 1) -- computes scalar score

### 3.5 Full Model Summary (`Im2Latex` in model.py)

```bash
Input image (B, 1, 64, 256)
        |
   ConvEncoder      -- 4x [Conv3x3 + BN + ReLU + Pool2x2]
        |            -- output: (B, 512, 4, 16)
        v
   RowEncoder       -- flatten columns + biLSTM
        |            -- output: (B, 16, 512)
        v
   Decoder           -- LSTM + Bahdanau attention
        |            -- generates tokens autoregressively
        v
LaTeX token sequence
```

## 4. Hyperparameters (config.py)

**Why each value was chosen:**

| Parameter               | Value                      | Why                                                                                                                                                                                                                                         |
| ----------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Batch size**          | 32                         | Balances GPU memory usage and gradient noise. Too large (128+) wastes memory and gives less noisy gradients (slower convergence). Too small (4-8) makes training unstable. 32 is a standard default that works well for image-to-seq tasks. |
| **Epochs**              | 40                         | Enough for the model to converge. The loss/BLEU curves flatten by ~30 epochs. More epochs would risk overfitting.                                                                                                                           |
| **Optimizer**           | Adam                       | Adam adapts the learning rate per-parameter using momentum. Converges faster than plain SGD for sequence models. Standard choice for encoder-decoder architectures.                                                                         |
| **Learning rate**       | 0.001                      | Default for Adam. The paper uses similar values. Too high (0.01) causes divergence, too low (0.0001) makes training very slow.                                                                                                              |
| **LR schedule**         | halved every 10 epochs     | After initial fast learning, smaller steps help fine-tune. Halving (gamma=0.5) is gentle enough to not kill learning. Every 10 epochs = 4 drops across 40 epochs (0.001 --> 0.0005 --> 0.00025 --> 0.000125).                               |
| **Gradient clipping**   | 5.0                        | LSTMs can have exploding gradients (especially early in training). Clipping at 5.0 prevents this without being so aggressive that it slows learning.                                                                                        |
| **Teacher forcing**     | 1.0 --> 0.6 (linear decay) | At start, feed ground truth tokens so the model learns faster. Gradually reduce to 60% so the model learns to recover from its own mistakes (exposure bias). Going below 0.5 makes early training unstable.                                 |
| **Max sequence length** | 200                        | Longest formula in the dataset. Prevents infinite decoding loops.                                                                                                                                                                           |
| **CNN filters**         | [64, 128, 256, 512]        | Doubling channels is standard (VGG-style). 512 at the end gives rich features. More would increase memory cost with diminishing returns.                                                                                                    |
| **Encoder hidden**      | 256                        | biLSTM with 256 gives 512-dim output (matching decoder). Larger would add parameters without clear benefit for 16-length sequences.                                                                                                         |
| **Embedding dim**       | 128                        | Standard for vocabularies in the hundreds. Larger (256+) would overparameterize for a small token vocabulary.                                                                                                                               |
| **Decoder hidden**      | 512                        | Must be large enough to model long sequences (up to 200 tokens). 512 is the standard in the reference paper.                                                                                                                                |
| **Attention dim**       | 256                        | Bottleneck dimension for computing scores. 256 gives enough capacity without being wasteful.                                                                                                                                                |
| **Decoder dropout**     | 0.2                        | Light regularization to prevent overfitting. 0.2 is standard for decoders. Encoder uses 0.1 (less needed since CNN already regularizes via BN).                                                                                             |
| **Beam width**          | 5                          | Top-5 candidates during beam search. Diminishing returns beyond 5. Width 1 = greedy. Width 10+ is slow with minimal improvement.                                                                                                            |

**Loss function:** Cross-entropy, ignoring PAD tokens (index 0) and the SOS position. This is standard for sequence generation -- it penalizes the model for each wrong token prediction.

## 5. Files -- What Each One Does

| File               | Purpose                                                                                                                                                                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`        | **All paths + hyperparameters in one place.** Change any setting here instead of digging through other files. Controls CNN size, LSTM dims, learning rate, batch size, attention on/off, etc.                                                                  |
| `vocab.py`         | **Builds the token vocabulary.** Reads `train_formulas.txt`, collects all unique tokens, assigns each an integer ID. Provides `encode()` (tokens --> IDs) and `decode()` (IDs --> tokens). Saves/loads to `vocab.pkl` so it only builds once.                  |
| `dataset.py`       | **PyTorch datasets + data loading.** `FormulaDataset` loads train/val image-formula pairs, applies preprocessing (resize, normalize, augment). `TestDataset` loads test images only (no labels). `collate_train` pads sequences to equal length in each batch. |
| `model.py`         | **The neural network.** Contains all four components: `ConvEncoder` (CNN), `RowEncoder` (biLSTM), `Attention` (Bahdanau), `Decoder` (LSTM). Also implements `greedy()` and `beam_decode()` inference methods.                                                  |
| `train.py`         | **Training loop.** Runs epochs, computes loss, validates, saves checkpoints, plots curves. Handles teacher forcing decay, learning rate scheduling, gradient clipping. Auto-resumes from checkpoint if one exists.                                             |
| `predict.py`       | **Generates test predictions.** Loads the best checkpoint, runs inference on test images using greedy or beam search, optionally applies post-processing. Outputs `test_formulas.txt`.                                                                         |
| `postprocess.py`   | **Cleans up generated LaTeX.** Fixes common decoder errors: unbalanced braces, empty arguments, double superscripts, repeated tokens (stuttering). Makes output actually compilable.                                                                           |
| `bleu_score.py`    | **BLEU metric** (provided). Computes n-gram overlap between predictions and ground truth.                                                                                                                                                                      |
| `edit_distance.py` | **Edit distance metric** (provided). Computes normalized Levenshtein distance between predictions and ground truth.                                                                                                                                            |
| `run_all.ipynb`    | **Colab notebook** to run the full pipeline step by step: setup, train, evaluate, visualize.                                                                                                                                                                   |

## 6. Metrics

**BLEU** (Bilingual Evaluation Understudy): Measures n-gram overlap between predicted and reference token sequences. Ranges from 0 to 1. Higher is better. We use 4-gram BLEU. A score of 0.80 means 80% of predicted n-grams match the reference.

**Edit Distance** (Normalized Levenshtein): Minimum number of token insertions, deletions, or substitutions needed to convert the prediction into the reference, divided by the length of the longer sequence. We report **accuracy = 1 - normalized_distance**. Higher is better.

## 7. Results

### Training Curves

![Loss](plots/loss.png)
![BLEU](plots/bleu.png)
![Edit Distance](plots/edit_distance.png)

### Visual Comparison

![Comparison](plots/comparison.png)

### Final Validation Numbers

<!-- TODO: Fill these numbers from your cell 10 and cell 14 output -->

| Metric                 | Value      |
| ---------------------- | ---------- |
| BLEU (4-gram)          | `0.831935` |
| Edit Distance Accuracy | `0.868808` |

attention to this point -> higher `BLEU` is better and lower the `Edit Distance` but in table its `Edit Distance Accuracy` so higher is better...

### have a look at this (i always say the truth!)

![alt text](./images/BLEU_DISTANCE.png)

## 8. Decoding Strategies

Two strategies implemented in `model.py`:

- **Greedy** (`model.greedy()`): At each step, pick the single highest-probability token. Fast (one forward pass per step), but can miss better sequences.

- **Beam search** (`model.beam_decode()`, width=5): Keep the top-5 partial sequences at each step and expand all of them. Picks the final sequence with the highest cumulative log-probability. Slower (5x more computation) but produces better output because it explores multiple possibilities.

## 9. Post-Processing (postprocess.py)

The decoder sometimes produces invalid LaTeX. These cleanup steps fix that:

| Step                    | What it fixes                         | Example                                     |
| ----------------------- | ------------------------------------- | ------------------------------------------- |
| **Stutter removal**     | Decoder repeats tokens 3+ times       | `\sum \sum \sum \sum` --> `\sum`            |
| **Brace balancing**     | Unmatched `{ } ( ) [ ]`               | `\frac { { a + b` --> `\frac { { a + b } }` |
| **Empty argument fix**  | Commands with empty `{}`              | `\frac { }` --> `\frac {~}`                 |
| **Double script merge** | Two superscripts/subscripts in a row  | `x ^ {a} ^ {b}` --> `x ^ { a ^ { b } }`     |
| **Math wrapping**       | Missing `$...$` delimiters (optional) | `x + y` --> `$ x + y $`                     |

### have a look at this pic:

![alt text](./images/POSTPROCCESS.png)

## 10. Attention vs No-Attention

> IMPORTANT : because of the restriction of Colab (it takes **10** hours to run on **T4**) i cant run the model twice so i have the numbers just with **attention** but the model works with both attention `True` or `False` you can change the `config.py` and try this at home:)

| Model             | BLEU                           | Edit Dist Accuracy             |
| ----------------- | ------------------------------ | ------------------------------ |
| Without attention | `PLS TRAIN THE MODEL YOURSELF` | `PLS TRAIN THE MODEL YOURSELF` |
| With attention    | `0.831935`                     | `0.868808`                     |

> **How to compare:**
>
> 1. In `config.py`, set `USE_ATTN = False`
> 2. Delete or rename the `checkpoints/` folder (so it trains fresh) -> because of the resume logic i have `its for restriction` if you hit train it start from the model that trained 40 epoch you need the rename or set the epoch higher
> 3. Run training again (cell 6 in notebook)
> 4. Run evaluation (cells 13-14) and fill the "Without attention" row
> 5. Set `USE_ATTN = True` back, and fill the "With attention" row from your current results

**Expected result:** Attention should give better scores because the decoder can look back at specific parts of the image at each step, instead of compressing everything into one hidden vector.

## 11. References

1. Deng, Yuntian, Anssi Kanervisto, and Alexander M. Rush. "What you get is what you see: A visual markup decompiler." arXiv:1609.04938 (2016).

2. Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." ICLR 2015.

3. A Simple Overview of RNN, LSTM and Attention Mechanism `MEDIUM`

---

## 12. Who am I ?

- **Student** : Hasan Aghaei
- **Professor** : Dr.Adeleh Bitarafan
