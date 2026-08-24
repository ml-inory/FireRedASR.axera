#!/usr/bin/env python3
"""生成 FSMN-VAD 校准集：fbank(80)+LFR(5)+CMVN -> [1,998,400] npy。

用法：python make_calib_fsmn.py --wav a.wav b.wav ... -o calib_data/fsmn_speech.tar.gz
建议用真实业务音频（含语音/静音/噪声），文件会按 10s 块切分。
"""
import argparse
import tarfile
import tempfile
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf


def load_cmvn(path):
    means, vars_ = [], []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
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
    return np.array(means, dtype=np.float64), np.array(vars_, dtype=np.float64)


def feats_for(chunk, means, vars_, max_len=998):
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
    if len(out) < max_len:
        out = np.pad(out, ((0, max_len - len(out)), (0, 0)))
    return out[:max_len][None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", nargs="+", required=True)
    ap.add_argument("--cmvn", default="am.mvn")
    ap.add_argument("-o", "--output", default="calib_data/fsmn_speech.tar.gz")
    args = ap.parse_args()
    means, vars_ = load_cmvn(args.cmvn)
    samples = []
    for wav in args.wavs:
        data, sr = sf.read(wav, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            x = np.arange(0, len(data), sr / 16000)
            data = np.interp(x, np.arange(len(data)), data).astype(np.float32)
        for st in range(0, len(data), 160000):
            chunk = data[st:st + 160000]
            if len(chunk) < 160000:
                chunk = np.pad(chunk, (0, 160000 - len(chunk)))
            samples.append(feats_for(chunk, means, vars_))
            print(wav, st / 16000)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, s in enumerate(samples):
            np.save(f"{td}/{i:04d}.npy", s)
        with tarfile.open(args.output, "w:gz") as t:
            for i in range(len(samples)):
                t.add(f"{td}/{i:04d}.npy", arcname=f"{i:04d}.npy")
    print(f"wrote {len(samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
