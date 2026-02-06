# model.py
# The full encoder-decoder network.
#
# Structure based on the paper "What You Get Is What You See" by Deng et al.:
#   1) CNN extracts a feature map from the image
#   2) Bidirectional LSTM reads the feature map columns (row encoder)
#   3) LSTM decoder generates tokens one at a time
#
# The attention module (Bahdanau-style) is optional -- controlled by USE_ATTN.

import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C


# ===========
#  CNN part
# ===========

class ConvEncoder(nn.Module):
    """Stack of Conv-BN-ReLU-Pool blocks."""

    def __init__(self):
        super().__init__()
        layers = []
        ch_in = C.IMG_CHANNELS
        for ch_out in C.CNN_FILTERS:
            layers += [
                nn.Conv2d(ch_in, ch_out, 3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
            ]
            ch_in = ch_out
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, H, W)  ->  (B, C', H', W')
        return self.backbone(x)


# ===============
#  Row encoder
# ===============

class RowEncoder(nn.Module):
    """
    Reads columns of the feature map as a sequence.
    Each column is flattened (channels * height) and fed into a biLSTM.
    """

    def __init__(self, n_channels, feat_h):
        super().__init__()
        inp = n_channels * feat_h
        self.lstm = nn.LSTM(
            inp, C.ENC_HIDDEN,
            num_layers=C.ENC_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=C.ENC_DROP if C.ENC_LAYERS > 1 else 0,
        )

    def forward(self, fmap):
        B, Ch, H, W = fmap.shape
        # treat each column as a timestep: (B, W, Ch*H)
        seq = fmap.permute(0, 3, 1, 2).reshape(B, W, Ch * H)
        out, states = self.lstm(seq)
        return out, states  # out: (B, W, ENC_HIDDEN*2)


# ==============
#  Attention
# ==============

class Attention(nn.Module):
    """Bahdanau (additive) attention."""

    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.We = nn.Linear(enc_dim, att_dim, bias=False)
        self.Wd = nn.Linear(dec_dim, att_dim, bias=False)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, enc_out, h_dec):
        # enc_out: (B, L, enc_dim),  h_dec: (B, dec_dim)
        e = self.We(enc_out)                     # (B, L, att)
        d = self.Wd(h_dec).unsqueeze(1)          # (B, 1, att)
        scores = self.v(torch.tanh(e + d))       # (B, L, 1)
        weights = F.softmax(scores.squeeze(2), 1)  # (B, L)
        ctx = (weights.unsqueeze(1) @ enc_out).squeeze(1)  # (B, enc_dim)
        return ctx, weights


# ============
#  Decoder
# ============

class Decoder(nn.Module):
    """LSTM decoder, optionally using attention."""

    def __init__(self, n_vocab, enc_dim, use_attn=True):
        super().__init__()
        self.n_vocab = n_vocab
        self.use_attn = use_attn
        self.enc_dim = enc_dim

        self.embed = nn.Embedding(n_vocab, C.EMBED_DIM, padding_idx=0)
        self.drop = nn.Dropout(C.DEC_DROP)

        rnn_inp = C.EMBED_DIM + (enc_dim if use_attn else 0)
        self.lstm = nn.LSTM(rnn_inp, C.DEC_HIDDEN,
                            C.DEC_LAYERS, batch_first=True)

        if use_attn:
            self.attn = Attention(enc_dim, C.DEC_HIDDEN, C.ATTN_DIM)
            self.out_proj = nn.Linear(C.DEC_HIDDEN + enc_dim, n_vocab)
        else:
            self.attn = None
            self.out_proj = nn.Linear(C.DEC_HIDDEN, n_vocab)

        # learnable init for the LSTM state, derived from encoder
        self.fc_h0 = nn.Linear(enc_dim, C.DEC_HIDDEN)
        self.fc_c0 = nn.Linear(enc_dim, C.DEC_HIDDEN)

    def init_hidden(self, enc_out):
        """Create initial (h, c) from the mean of encoder outputs."""
        avg = enc_out.mean(dim=1)
        h = torch.tanh(self.fc_h0(avg)).unsqueeze(
            0).expand(C.DEC_LAYERS, -1, -1).contiguous()
        c = torch.tanh(self.fc_c0(avg)).unsqueeze(
            0).expand(C.DEC_LAYERS, -1, -1).contiguous()
        return h, c

    def step(self, tok, hidden, enc_out):
        """One decoding step.  tok: (B,)"""
        emb = self.drop(self.embed(tok))       # (B, EMBED)
        h_last = hidden[0][-1]                 # (B, DEC_HIDDEN)

        ctx = None
        aw = None
        if self.use_attn:
            ctx, aw = self.attn(enc_out, h_last)
            rnn_in = torch.cat([emb, ctx], 1)
        else:
            rnn_in = emb

        rnn_in = rnn_in.unsqueeze(1)           # (B, 1, *)
        out, hidden = self.lstm(rnn_in, hidden)
        out = out.squeeze(1)

        if self.use_attn:
            logits = self.out_proj(torch.cat([out, ctx], 1))
        else:
            logits = self.out_proj(out)
        return logits, hidden, aw

    def forward(self, enc_out, targets, tf_ratio=1.0):
        """Full sequence decoding with teacher forcing."""
        B, T = targets.shape
        hidden = self.init_hidden(enc_out)

        all_logits = torch.zeros(B, T, self.n_vocab, device=targets.device)
        inp = targets[:, 0]  # should be SOS

        for t in range(1, T):
            logits, hidden, _ = self.step(inp, hidden, enc_out)
            all_logits[:, t] = logits
            # teacher forcing decision
            if torch.rand(1).item() < tf_ratio:
                inp = targets[:, t]
            else:
                inp = logits.argmax(1)
        return all_logits


# ==================
#  Combined model
# ==================

class Im2Latex(nn.Module):
    """End-to-end: image -> CNN -> row LSTM -> decoder -> tokens."""

    def __init__(self, vocab_size):
        super().__init__()
        self.cnn = ConvEncoder()

        # figure out spatial dims after pooling
        n_pools = len(C.CNN_FILTERS)
        self.fh = C.IMG_H // (2 ** n_pools)
        self.fw = C.IMG_W // (2 ** n_pools)
        fc = C.CNN_FILTERS[-1]

        self.encoder = RowEncoder(fc, self.fh)
        enc_dim = C.ENC_HIDDEN * 2  # bidirectional
        self.decoder = Decoder(vocab_size, enc_dim, use_attn=C.USE_ATTN)

    def forward(self, imgs, targets, tf_ratio=1.0):
        features = self.cnn(imgs)
        enc, _ = self.encoder(features)
        return self.decoder(enc, targets, tf_ratio)

    @torch.no_grad()
    def greedy(self, imgs, sos, eos, max_len=None):
        """Greedy decoding -- pick argmax at each step."""
        max_len = max_len or C.MAX_SEQ
        features = self.cnn(imgs)
        enc, _ = self.encoder(features)
        B = imgs.size(0)
        dev = imgs.device

        hid = self.decoder.init_hidden(enc)
        tok = torch.full((B,), sos, dtype=torch.long, device=dev)
        done = [False] * B
        seqs = [[] for _ in range(B)]

        for _ in range(max_len):
            logits, hid, _ = self.decoder.step(tok, hid, enc)
            pred = logits.argmax(1)
            for b in range(B):
                p = pred[b].item()
                if done[b]:
                    continue
                if p == eos:
                    done[b] = True
                else:
                    seqs[b].append(p)
            if all(done):
                break
            tok = pred
        return seqs

    @torch.no_grad()
    def beam_decode(self, imgs, sos, eos, beam_k=None, max_len=None):
        """
        Beam search.  Does one image at a time inside the batch
        (simpler to implement, still fast enough for inference).
        """
        beam_k = beam_k or C.BEAM_K
        max_len = max_len or C.MAX_SEQ

        features = self.cnn(imgs)
        enc, _ = self.encoder(features)
        B = imgs.size(0)
        dev = imgs.device
        results = []

        for b in range(B):
            enc_b = enc[b:b+1]  # (1, L, dim)
            h0 = self.decoder.init_hidden(enc_b)
            start = torch.tensor([sos], device=dev)

            # each beam: (token_list, cumul_log_prob, hidden, finished)
            beams = [([], 0.0, h0, False)]

            for _ in range(max_len):
                new_beams = []
                for seq, score, hid, fin in beams:
                    if fin:
                        new_beams.append((seq, score, hid, True))
                        continue
                    inp = start if len(seq) == 0 else torch.tensor(
                        [seq[-1]], device=dev)
                    logits, new_hid, _ = self.decoder.step(inp, hid, enc_b)
                    lp = F.log_softmax(logits, 1).squeeze(0)
                    vals, ids = lp.topk(beam_k)
                    for k in range(beam_k):
                        tid = ids[k].item()
                        ns = score + vals[k].item()
                        if tid == eos:
                            new_beams.append((seq, ns, new_hid, True))
                        else:
                            new_beams.append((seq + [tid], ns, new_hid, False))
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_k]
                if all(fin for _, _, _, fin in beams):
                    break

            results.append(beams[0][0])
        return results
