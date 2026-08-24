"""FireRedASR-AED numpy+pyaxengine ASR 核心（板端，无 torch/onnxruntime）。"""
import math
from pathlib import Path

import kaldiio
import kaldi_native_fbank as knf
import numpy as np

from fsmn_vad_post import vad_segments


def _fbank_cmvn(pcm_int16, means, invstds, max_feat_len):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, pcm_int16.tolist())
    n = fbank.num_frames_ready
    feat_len = min(n, max_feat_len)
    feats = np.zeros((max_feat_len, 80), dtype=np.float32)
    for i in range(feat_len):
        frame = np.array(fbank.get_frame(i), dtype=np.float32)
        feats[i] = (frame - means) * invstds
    return feats[None], feat_len


def load_cmvn(path):
    stats = kaldiio.load_mat(path)
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    means = stats[0, :dim] / count
    var = np.maximum(stats[1, :dim] / count - means * means, 1e-20)
    return np.asarray(means, dtype=np.float32), np.asarray(1.0 / np.sqrt(var), dtype=np.float32)


def load_dict(path):
    id2word = {}
    for line in open(path, encoding="utf-8"):
        p = line.strip().split()
        if len(p) >= 2:
            id2word[int(p[1])] = p[0]
    return id2word


def load_fsmn_cmvn(path):
    means, vars_ = [], []
    lines = open(path, encoding="utf-8").readlines()
    for i, line in enumerate(lines):
        items = line.split()
        if not items:
            continue
        if items[0] == "<AddShift>":
            j = i + 1
            if j < len(lines) and lines[j].split()[0] == "<LearnRateCoef>":
                means = [float(x) for x in lines[j].split()[3:-1]]
        elif items[0] == "<Rescale>":
            j = i + 1
            if j < len(lines) and lines[j].split()[0] == "<LearnRateCoef>":
                vars_ = [float(x) for x in lines[j].split()[3:-1]]
    return np.asarray(means, dtype=np.float64), np.asarray(vars_, dtype=np.float64)


def _fsmn_feats(chunk, means, vars_):
    """FSMN-VAD 前端：fbank(80) + LFR(5) + CMVN，输出 [1,998,400] float32。"""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.mel_opts.num_bins = 80
    opts.energy_floor = 0.0
    opts.frame_opts.snip_edges = True
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, (chunk * 32768).tolist())
    n = fbank.num_frames_ready
    feat = np.empty((n, 80), dtype=np.float32)
    for i in range(n):
        feat[i] = fbank.get_frame(i)
    left = np.tile(feat[0], (2, 1))
    feat = np.vstack((left, feat))
    out = np.stack([
        feat[i:i + 5].reshape(-1) for i in range(len(feat) - 4)
    ]).astype(np.float32)
    out = (out + means) * vars_
    if len(out) < 998:
        out = np.pad(out, ((0, 998 - len(out)), (0, 0)))
    return np.asarray(out[None], dtype=np.float32)


class FireredASR:
    def __init__(self, encoder, decoder, fsmn_vad, fsmn_cmvn,
                 cmvn, dict_path, pe, max_dur=10, max_steps=128,
                 fsmn_threshold=0.6, vad_min_speech_ms=1000):
        import axengine as axe

        self.enc = axe.InferenceSession(encoder)
        self.dec = axe.InferenceSession(decoder)
        self.fsmn = axe.InferenceSession(fsmn_vad)
        self.fsmn_means, self.fsmn_vars = load_fsmn_cmvn(fsmn_cmvn)
        self.fsmn_threshold = fsmn_threshold
        self.vad_min_speech_ms = vad_min_speech_ms
        self.means, self.invstds = load_cmvn(cmvn)
        self.id2word = load_dict(dict_path)
        self.pe = np.ascontiguousarray(np.load(pe), dtype=np.float32)
        self.max_feat_len = math.floor((max_dur * 16000 - 400) / 160) + 1
        self.max_chunk_samples = max_dur * 16000
        self.max_steps = max_steps
        self.cache_len = self.dec.get_inputs()[1].shape[2]
        self.vocab = self.dec.get_outputs()[0].shape[2]

    def _detok(self, ids):
        s = "".join(self.id2word.get(i, "") for i in ids if i != 4)
        return s.replace("▁", " ").strip()

    def _transcribe_chunk(self, chunk):
        pcm16 = (np.clip(chunk, -1, 1) * 32768).astype(np.int16)
        feats, length = _fbank_cmvn(pcm16, self.means, self.invstds, self.max_feat_len)
        cross_k, cross_v, cross_mask = self.enc.run(
            None, {"encoder_input": feats,
                   "encoder_input_lengths": np.array([length], dtype=np.int32)})
        n_layers, hidden = 16, 1280
        k_cache = np.zeros((n_layers, 1, self.cache_len, hidden), dtype=np.float32)
        v_cache = np.zeros((n_layers, 1, self.cache_len, hidden), dtype=np.float32)
        tokens = np.array([[3]], dtype=np.int32)
        masks = []
        for off in range(self.cache_len):
            m = np.zeros((1, 1, self.cache_len), dtype=np.float32)
            m[:, :, : self.cache_len - off - 1] = -np.inf
            masks.append(m)
        ids = []
        for off in range(self.max_steps):
            logits, k_cache, v_cache = self.dec.run(
                None, {
                    "tokens": tokens,
                    "in_n_layer_self_k_cache": k_cache,
                    "in_n_layer_self_v_cache": v_cache,
                    "n_layer_cross_k": cross_k,
                    "n_layer_cross_v": cross_v,
                    "pe": self.pe[off],
                    "self_attn_mask": masks[off],
                    "cross_attn_mask": cross_mask,
                })
            nxt = int(np.argmax(logits[0, 0]))
            tokens[0, 0] = nxt
            ids.append(nxt)
            if nxt == 4:
                break
        return ids

    def transcribe(self, wav_path):
        import soundfile as sf
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)  # 立体声/多声道混单声道
        if sr != 16000:
            x = np.arange(0, len(data), sr / 16000)
            data = np.interp(x, np.arange(len(data)), data).astype(np.float32)

        all_tokens = []
        step = self.max_chunk_samples
        for start in range(0, len(data), step):
            chunk = data[start:min(start + step, len(data))]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            feats = _fsmn_feats(chunk, self.fsmn_means, self.fsmn_vars)
            logits = np.asarray(self.fsmn.run(None, {"speech": feats})[0])
            segs = vad_segments(logits, chunk[None], self.fsmn_threshold)
            if sum(e - s for s, e in segs) < self.vad_min_speech_ms:
                continue  # 静音/噪声块不送 ASR
            all_tokens.extend(self._transcribe_chunk(
                data[start:min(start + step, len(data))]))
        return self._detok(all_tokens) if all_tokens else ""
