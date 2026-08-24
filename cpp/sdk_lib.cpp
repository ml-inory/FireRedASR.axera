#include "sdk_lib.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include "Encoder.hpp"
#include "DecoderLoop.hpp"
#include "sdk_common.hpp"

struct FireredSdk::Impl {
    Encoder enc;
    DecoderLoop dec;
    FsmnVad fsmn_vad;
    std::vector<float> means, invstds;
    std::vector<float> fsmn_means, fsmn_vars;
    std::vector<std::string> id2word;
    std::vector<float> pe;
    int max_feat_len = 998;
    int max_chunk_samples = 160000;
    int max_steps = 128;
    int fsmn_min_speech_ms = 1000;
    bool inited = false;
};

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

FireredSdk::~FireredSdk() { delete impl_; }

bool FireredSdk::Init(const char* encoder, const char* decoder,
                      const char* fsmn_vad, const char* fsmn_cmvn,
                      const char* cmvn, const char* dict, const char* pe_path,
                      int max_dur_s, int max_steps, int vad_min_speech_ms) {
    if (!impl_) impl_ = new Impl();
    if (AX_SYS_Init() != 0) return false;
    AX_ENGINE_NPU_ATTR_T attr;
    memset(&attr, 0, sizeof(attr));
    attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    if (AX_ENGINE_Init(&attr) != 0) return false;
    if (impl_->enc.Init(encoder) != 0) return false;
    if (impl_->dec.Init(decoder) != 0) return false;
    if (!impl_->fsmn_vad.Init(fsmn_vad)) return false;
    if (!load_fsmn_cmvn(fsmn_cmvn, impl_->fsmn_means, impl_->fsmn_vars)) return false;
    impl_->fsmn_min_speech_ms = vad_min_speech_ms > 0 ? vad_min_speech_ms : 1000;
    if (!load_cmvn(cmvn, impl_->means, impl_->invstds)) return false;
    if (!load_dict(dict, impl_->id2word)) return false;
    FILE* pf = fopen(pe_path, "rb");
    if (!pf) return false;
    impl_->pe.resize(5000 * 1280);
    size_t rd = fread(impl_->pe.data(), 4, impl_->pe.size(), pf);
    fclose(pf);
    if (rd != impl_->pe.size()) return false;
    impl_->max_feat_len = (int)(((max_dur_s * 16000 - 400) / 160) + 1);
    impl_->max_chunk_samples = max_dur_s * 16000;
    impl_->max_steps = max_steps;
    impl_->inited = true;
    return true;
}

std::string FireredSdk::Transcribe(const char* wav_path,
                                   double* vad_ms, double* enc_ms, double* dec_ms) {
    if (!impl_ || !impl_->inited) return "";
    Impl& S = *impl_;

    double t_vad0 = now_ms();
    WavData wav;
    if (!read_wav_pcm16(wav_path, wav)) return "";
    wav.samples = resample_to_16k(wav.samples, wav.sample_rate);
    wav.sample_rate = 16000;
    std::vector<float> wav_norm(wav.samples.size());
    for (size_t i = 0; i < wav.samples.size(); i++) wav_norm[i] = wav.samples[i] / 32768.0f;

    // FSMN-VAD 块级 gate：10s 块内检出语音 >= vad_min_speech_ms 才整块送 ASR
    std::vector<std::vector<float>> chunks;
        int step = S.max_chunk_samples;
        for (size_t start = 0; start < wav_norm.size(); start += step) {
            std::vector<float> chunk(wav_norm.begin() + start,
                                     wav_norm.begin() + std::min(start + step, wav_norm.size()));
            if (chunk.size() < (size_t)step) chunk.resize(step, 0.0f);
            auto feats = compute_fsmn_feats(chunk, S.fsmn_means, S.fsmn_vars);
            std::vector<float> logits;
            S.fsmn_vad.Run(feats.data(), logits);
            int seg_ms = fsmn_segment_ms(logits);
            if (seg_ms < S.fsmn_min_speech_ms) continue;
            size_t real_n = std::min(start + step, wav_norm.size()) - start;
            chunks.push_back(std::vector<float>(wav_norm.begin() + start,
                                                wav_norm.begin() + start + real_n));
        }
        if (chunks.empty()) {
            if (vad_ms) *vad_ms = now_ms() - t_vad0;
            return "";
        }
        double t_vad1 = now_ms();
        if (vad_ms) *vad_ms = t_vad1 - t_vad0;

        std::vector<std::vector<float>> feats(chunks.size());
        std::vector<int> lengths(chunks.size());
        auto compute_feat = [&](size_t i) -> std::vector<float> {
            std::vector<float> pcm16(chunks[i].size());
            for (size_t j = 0; j < chunks[i].size(); j++)
                pcm16[j] = std::max(-1.0f, std::min(1.0f, chunks[i][j])) * 32768.0f;
            int len = 0;
            auto f = compute_fbank_cmvn(pcm16, S.means, S.invstds, S.max_feat_len, len);
            lengths[i] = len;
            return f;
        };
        feats[0] = compute_feat(0);

        std::vector<int> all_tokens;
        double enc_total = 0, dec_total = 0;
        for (size_t ci = 0; ci < chunks.size(); ci++) {
            std::thread feat_thread;
            if (ci + 1 < chunks.size()) {
                feat_thread = std::thread([&, ci]() {
                    feats[ci + 1] = compute_feat(ci + 1);
                });
            }
            double t0 = now_ms();
            S.enc.SetInput(feats[ci].data(), 0);
            S.enc.SetInput(&lengths[ci], 1);
            S.enc.Run();
            double t1 = now_ms();
            enc_total += t1 - t0;

            int cross_k_n = S.enc.GetOutputSize(0) / 4;
            int cross_v_n = S.enc.GetOutputSize(1) / 4;
            int mask_n = S.enc.GetOutputSize(2) / 4;
            std::vector<float> cross_k(cross_k_n), cross_v(cross_v_n), cross_mask(mask_n);
            S.enc.GetOutput(cross_k.data(), 0);
            S.enc.GetOutput(cross_v.data(), 1);
            S.enc.GetOutput(cross_mask.data(), 2);

            int hidden = 1280;
            int cache_len = S.dec.GetInputSize(1) / 4 / 16 / hidden;
            int n_layers = S.dec.GetInputSize(1) / 4 / cache_len / hidden;
            int vocab = S.dec.GetOutputSize(0) / 4;
            std::vector<float> k_cache((size_t)n_layers * cache_len * hidden, 0.0f);
            std::vector<float> v_cache((size_t)n_layers * cache_len * hidden, 0.0f);
            std::vector<int32_t> tokens = {3};
            std::vector<float> self_mask(cache_len, 0.0f);
            std::vector<float> pe_step(hidden);
            std::vector<float> logits(vocab);

            S.dec.SetInput(cross_k.data(), 3);
            S.dec.SetInput(cross_v.data(), 4);
            S.dec.SetInput(cross_mask.data(), 7);
            S.dec.SetInput(k_cache.data(), 1);
            S.dec.SetInput(v_cache.data(), 2);

            double s = now_ms();
            for (int off = 0; off < S.max_steps; off++) {
                std::fill(self_mask.begin(), self_mask.end(), 0.0f);
                std::fill(self_mask.begin(), self_mask.begin() + (cache_len - off - 1), -INFINITY);
                memcpy(pe_step.data(), S.pe.data() + (size_t)off * hidden, hidden * sizeof(float));
                S.dec.SetInput(tokens.data(), 0);
                S.dec.SetInput(pe_step.data(), 5);
                S.dec.SetInput(self_mask.data(), 6);
                S.dec.Run();
                S.dec.GetOutput(logits.data(), 0);
                S.dec.SetInputFromOutput(1, 1);
                S.dec.SetInputFromOutput(2, 2);
                int best = 0; float bestv = logits[0];
                for (int i = 1; i < vocab; i++) if (logits[i] > bestv) { bestv = logits[i]; best = i; }
                tokens[0] = best;
                all_tokens.push_back(best);
                if (best == 4) break;
            }
            double e = now_ms();
            dec_total += e - s;
            if (feat_thread.joinable()) feat_thread.join();
        }
        if (enc_ms) *enc_ms = enc_total;
        if (dec_ms) *dec_ms = dec_total;
        return detokenize(all_tokens, S.id2word);
}

// ---- C API ----
FireredHandle firered_create(const char* encoder, const char* decoder,
                             const char* fsmn_vad, const char* fsmn_cmvn,
                             const char* cmvn, const char* dict,
                             const char* pe_path, int max_dur_s, int max_steps,
                             int vad_min_speech_ms) {
    FireredSdk* sdk = new FireredSdk();
    if (!sdk->Init(encoder, decoder, fsmn_vad, fsmn_cmvn,
                   cmvn, dict, pe_path, max_dur_s, max_steps,
                   vad_min_speech_ms)) {
        delete sdk;
        return nullptr;
    }
    return sdk;
}

int firered_transcribe(FireredHandle h, const char* wav, char* out, int out_size,
                       double* vad_ms, double* enc_ms, double* dec_ms) {
    if (!h || !out || out_size <= 0) return -1;
    std::string text = ((FireredSdk*)h)->Transcribe(wav, vad_ms, enc_ms, dec_ms);
    if ((int)text.size() >= out_size) return -2;
    memcpy(out, text.c_str(), text.size() + 1);
    return 0;
}

void firered_destroy(FireredHandle h) { delete (FireredSdk*)h; }
