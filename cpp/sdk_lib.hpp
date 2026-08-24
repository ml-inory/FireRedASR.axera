#pragma once

#include <string>
#include <vector>

// FireRedASR-AED AX650N 端到端 SDK（板端 VAD + fbank + ASR）
class FireredSdk {
public:
    FireredSdk() = default;
    ~FireredSdk();

    // 初始化：encoder/decoder、FSMN-VAD axmodel + am.mvn、ASR cmvn、词典、pe
    bool Init(const char* encoder, const char* decoder,
              const char* fsmn_vad, const char* fsmn_cmvn,
              const char* cmvn, const char* dict, const char* pe_path,
              int max_dur_s, int max_steps, int vad_min_speech_ms = 1000);

    // 识别 wav，输出文本；可选返回各阶段耗时（ms）
    std::string Transcribe(const char* wav_path,
                           double* vad_ms = nullptr,
                           double* enc_ms = nullptr,
                           double* dec_ms = nullptr);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// 默认 VAD 最小语音段（ms）

// 简易 C API（供其他语言/框架调用）
#ifdef __cplusplus
extern "C" {
#endif
typedef void* FireredHandle;
FireredHandle firered_create(const char* encoder, const char* decoder,
                             const char* fsmn_vad, const char* fsmn_cmvn,
                             const char* cmvn, const char* dict,
                             const char* pe_path, int max_dur_s, int max_steps,
                             int vad_min_speech_ms = 1000);
int firered_transcribe(FireredHandle h, const char* wav, char* out, int out_size,
                       double* vad_ms, double* enc_ms, double* dec_ms);
void firered_destroy(FireredHandle h);
#ifdef __cplusplus
}
#endif
