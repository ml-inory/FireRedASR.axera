#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <kaldi-native-fbank/csrc/online-feature.h>

#include "EngineWrapper.hpp"

// ---------- WAV（PCM16 mono，与 kaldiio 读取的 int16 原始值对齐） ----------
struct WavData {
    int sample_rate = 16000;
    std::vector<float> samples;  // int16 原值转 float（不做归一化，与 Python 管线一致）
};

static bool read_wav_pcm16(const std::string& path, WavData& wav) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char hdr[12];
    f.read(hdr, 12);
    if (memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) return false;
    int16_t channels = 0, bits = 0;
    int32_t sr = 0;
    bool found_data = false;
    while (f) {
        char ck[8];
        f.read(ck, 8);
        if (!f) break;
        uint32_t sz;
        memcpy(&sz, ck + 4, 4);
        if (!memcmp(ck, "fmt ", 4)) {
            std::vector<char> buf(sz);
            f.read(buf.data(), sz);
            if (sz >= 16) {
                memcpy(&channels, buf.data() + 2, 2);
                memcpy(&sr, buf.data() + 4, 4);
                memcpy(&bits, buf.data() + 14, 2);
            }
            if (sz % 2) f.seekg(1, std::ios::cur);
        } else if (!memcmp(ck, "data", 4)) {
            found_data = true;
            int32_t n = sz / 2;
            std::vector<int16_t> pcm(n);
            f.read((char*)pcm.data(), sz);
            wav.sample_rate = sr;
            int nch = channels > 0 ? channels : 1;
            int frames = n / nch;
            wav.samples.resize(frames);
            for (int i = 0; i < frames; i++) {
                int32_t acc = 0;
                for (int c = 0; c < nch; c++) acc += pcm[i * nch + c];
                wav.samples[i] = (float)(acc / nch);  // 多声道混单声道（PCM16）
            }
            break;
        } else {
            f.seekg(sz + (sz % 2), std::ios::cur);
        }
    }
    return found_data && channels >= 1;
}

// ---------- 重采样到 16k（线性插值，与 Python 基准一致） ----------
static std::vector<float> resample_to_16k(const std::vector<float>& in, int sr_in) {
    if (sr_in == 16000) return in;
    std::vector<float> out;
    double step = (double)sr_in / 16000.0;
    int n_out = (int)(in.size() / step);
    out.reserve(n_out);
    for (int i = 0; i < n_out; i++) {
        double pos = i * step;
        int i0 = (int)pos;
        int i1 = std::min(i0 + 1, (int)in.size() - 1);
        double frac = pos - i0;
        out.push_back((float)(in[i0] * (1.0 - frac) + in[i1] * frac));
    }
    return out;
}

// ---------- CMVN（kaldi binary DM 矩阵） ----------
static bool load_cmvn(const std::string& path, std::vector<float>& means, std::vector<float>& invstds) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char b0, b1;
    f.read(&b0, 1); f.read(&b1, 1);
    if (b0 != 0 || b1 != 'B') return false;
    char tok[3];
    f.read(tok, 3);  // "DM " / "FM "
    char m1;
    f.read(&m1, 1);
    if (m1 != 4) return false;
    int32_t rows = 0;
    f.read((char*)&rows, 4);
    f.read(&m1, 1);
    if (m1 != 4) return false;
    int32_t cols = 0;
    f.read((char*)&cols, 4);
    int is_double = (tok[0] == 'D');
    std::vector<double> data((size_t)rows * cols);
    if (is_double) {
        f.read((char*)data.data(), data.size() * 8);
    } else {
        std::vector<float> tmp((size_t)rows * cols);
        f.read((char*)tmp.data(), tmp.size() * 4);
        for (size_t i = 0; i < tmp.size(); i++) data[i] = tmp[i];
    }
    if (rows != 2 || cols != 81) return false;
    int dim = 80;
    double count = data[cols - 1];  // data[0][80]
    means.resize(dim);
    invstds.resize(dim);
    for (int d = 0; d < dim; d++) {
        double mean = data[d] / count;
        double var = data[cols + d] / count - mean * mean;
        if (var < 1e-20) var = 1e-20;
        means[d] = (float)mean;
        invstds[d] = (float)(1.0 / std::sqrt(var));
    }
    return true;
}

// ---------- FBank（kaldi-native-fbank，与 Python 参数一致） ----------
static std::vector<float> compute_fbank_cmvn(const std::vector<float>& pcm_int16,
                                              const std::vector<float>& means,
                                              const std::vector<float>& invstds,
                                              int max_feat_len,
                                              int& feat_len) {
    knf::FbankOptions opts;
    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.snip_edges = true;
    opts.mel_opts.num_bins = 80;
    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(16000, pcm_int16.data(), (int32_t)pcm_int16.size());
    fbank.InputFinished();
    int n = fbank.NumFramesReady();
    feat_len = std::min(n, max_feat_len);
    std::vector<float> feats((size_t)max_feat_len * 80, 0.0f);
    for (int i = 0; i < feat_len; i++) {
        const float* frame = fbank.GetFrame(i);
        for (int j = 0; j < 80; j++) {
            feats[(size_t)i * 80 + j] = (frame[j] - means[j]) * invstds[j];
        }
    }
    return feats;
}

// ---------- FSMN-VAD（fsmnvad-offline 10s axmodel + FunASR 块级端点后处理） ----------
static bool load_fsmn_cmvn(const std::string& path,
                           std::vector<float>& means,
                           std::vector<float>& vars) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::vector<float> means_tmp, vars_tmp;
    auto parse_coef_line = [](const std::string& s) -> std::vector<float> {
        std::vector<std::string> tok;
        std::istringstream iss(s);
        std::string t;
        while (iss >> t) tok.push_back(t);
        // 期望: <LearnRateCoef> 1 [ n0 n1 ... nN ]
        std::vector<float> vals;
        for (size_t i = 3; i + 1 < tok.size(); i++)
            vals.push_back(std::stof(tok[i]));
        return vals;
    };
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string tok;
        iss >> tok;
        if (tok == "<AddShift>") {
            std::string line2;
            if (!std::getline(f, line2)) break;
            means_tmp = parse_coef_line(line2);
        } else if (tok == "<Rescale>") {
            std::string line2;
            if (!std::getline(f, line2)) break;
            vars_tmp = parse_coef_line(line2);
        }
    }
    if (means_tmp.empty() || vars_tmp.empty()) return false;
    means = means_tmp;
    vars = vars_tmp;
    return true;
}

static std::vector<float> compute_fsmn_feats(const std::vector<float>& chunk_01,
                                              const std::vector<float>& fsmn_means,
                                              const std::vector<float>& fsmn_vars) {
    // 输入 float -1~1，pad/截断到 10s（160000 样本）
    const int MAXN = 160000;
    std::vector<float> pcm(MAXN, 0.0f);
    size_t n = std::min<size_t>(chunk_01.size(), MAXN);
    for (size_t i = 0; i < n; i++) pcm[i] = chunk_01[i] * 32768.0f;

    knf::FbankOptions opts;
    opts.frame_opts.dither = 0.0f;
    opts.frame_opts.window_type = "hamming";
    opts.frame_opts.frame_shift_ms = 10.0f;
    opts.frame_opts.frame_length_ms = 25.0f;
    opts.mel_opts.num_bins = 80;
    opts.energy_floor = 0.0f;
    opts.frame_opts.snip_edges = true;
    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(16000, pcm.data(), (int32_t)pcm.size());
    fbank.InputFinished();
    int T = std::min(fbank.NumFramesReady(), 998);
    std::vector<float> feat((size_t)T * 80);
    for (int i = 0; i < T; i++) {
        const float* fr = fbank.GetFrame(i);
        memcpy(feat.data() + (size_t)i * 80, fr, 80 * sizeof(float));
    }
    // LFR m=5 n=1：前 2 帧复制首帧
    const int D = 80, L = 5;
    std::vector<float> padded((size_t)(T + 2) * D);
    for (int j = 0; j < D; j++) padded[j] = feat[j];
    for (int j = 0; j < D; j++) padded[D + j] = feat[j];
    memcpy(padded.data() + 2 * D, feat.data(), (size_t)T * D * sizeof(float));
    std::vector<float> out((size_t)T * (D * L), 0.0f);
    for (int i = 0; i < T; i++) {
        float* dst = out.data() + (size_t)i * (D * L);
        const float* src = padded.data() + (size_t)i * D;
        for (int k = 0; k < L; k++)
            for (int j = 0; j < D; j++)
                dst[k * D + j] = src[k * D + j];
    }
    for (size_t i = 0; i < out.size(); i++)
        out[i] = (out[i] + fsmn_means[i % D]) * fsmn_vars[i % D];
    if (T < 998) {
        std::vector<float> padded_out((size_t)998 * D * L, 0.0f);
        memcpy(padded_out.data(), out.data(), out.size() * sizeof(float));
        out = std::move(padded_out);
    }
    return out;  // 998*400
}

class FsmnVad {
public:
    bool Init(const char* path) {
        return vad_.Init(path) == 0;
    }
    bool Run(const float* feats, std::vector<float>& logits) {
        vad_.SetInput((void*)feats, 0);
        int ret = vad_.Run();
        if (ret != 0) return false;
        logits.resize(vad_.GetOutputSize(0) / 4);
        vad_.GetOutput(logits.data(), 0);
        return true;
    }
private:
    EngineWrapper vad_;
};

// 块级端点统计：与 Python fsmn_vad_post 的块级 gate 对齐（简化窗口 + 扩展）
static int fsmn_segment_ms(const std::vector<float>& logits, float threshold = 0.6f) {
    const int T = 998;
    const int WIN = 20;      // 200ms
    const int ON = 15;       // 150ms
    std::vector<int> st(T);
    for (int t = 0; t < T; t++) {
        float p_sil = logits[(size_t)t * 248];
        float p_sp = 1.0f - p_sil;
        st[t] = (p_sp >= p_sil + threshold) ? 1 : 0;
    }
    int win_sum = 0;
    std::vector<int> win(WIN, 0);
    int pos = 0;
    int state = 0;           // 0=sil, 1=speech
    int start = 0;
    std::vector<std::pair<int,int>> segs;
    for (int t = 0; t < T; t++) {
        win_sum += st[t] - win[pos];
        win[pos] = st[t];
        pos = (pos + 1) % WIN;
        if (state == 0 && win_sum >= ON) {
            state = 1;
            start = std::max(0, t - WIN + 1 - 20);  // lookback 200ms
        } else if (state == 1 && win_sum <= ON) {
            int end = std::min(T - 1, t + 10);      // lookahead 100ms
            if (end - start >= ON) segs.push_back({start, end});
            state = 0;
        }
    }
    if (state == 1) {
        int end = std::min(T - 1, T - 1 + 10);
        if (end - start >= ON) segs.push_back({start, end});
    }
    // 合并相邻/重叠段（gap <= 200ms）
    std::vector<std::pair<int,int>> merged;
    for (auto& s : segs) {
        if (merged.empty() || s.first > merged.back().second + 20) {
            merged.push_back(s);
        } else {
            merged.back().second = std::max(merged.back().second, s.second);
        }
    }
    int total = 0;
    for (auto& s : merged) total += (s.second - s.first) * 10;
    return total;
}

// ---------- 词典 detokenize ----------
static bool load_dict(const std::string& path, std::vector<std::string>& id2word) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        size_t sp = line.find_first_of(" \t");
        if (sp == std::string::npos) continue;
        int id = atoi(line.c_str() + sp + 1);
        std::string w = line.substr(0, sp);
        if (w == "<space>") w = " ";
        if ((int)id2word.size() <= id) id2word.resize(id + 1);
        id2word[id] = w;
    }
    return true;
}

static std::string detokenize(const std::vector<int>& ids, const std::vector<std::string>& id2word) {
    std::string s;
    for (int id : ids) {
        if (id == 4) continue;  // <eos>
        if (id >= 0 && id < (int)id2word.size() && !id2word[id].empty())
            s += id2word[id];
    }
    std::string needle = "\xe2\x96\x81";
    size_t pos = 0;
    while ((pos = s.find(needle)) != std::string::npos) {
        s.replace(pos, needle.size(), " ");
    }
    std::string out = s;
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}
