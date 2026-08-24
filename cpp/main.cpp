// FireRedASR-AED AX650N C++ 基准：encoder + decoder_loop 贪心解码，真实输入。
// 输入：encoder_input.bin / encoder_input_lengths.bin / pe.bin（与 Python 基准一致）
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include "EngineWrapper.hpp"
#include "Encoder.hpp"
#include "DecoderLoop.hpp"

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool read_file(const char* path, std::vector<float>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n / sizeof(float));
    size_t rd = fread(out.data(), 1, n, f);
    fclose(f);
    return rd == (size_t)n;
}

static int32_t read_int32_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int32_t v = 0;
    size_t rd = fread(&v, 1, sizeof(v), f);
    fclose(f);
    return rd == sizeof(v) ? v : 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 5) {
        printf("usage: %s <encoder.axmodel> <decoder_loop.axmodel> <input_dir> <max_steps>\n", argv[0]);
        return 1;
    }
    const char* enc_path = argv[1];
    const char* dec_path = argv[2];
    std::string in_dir = argv[3];
    int max_steps = atoi(argv[4]);

    int ret = AX_SYS_Init();
    if (ret != 0) { printf("AX_SYS_Init fail 0x%x\n", ret); return 1; }
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) { printf("AX_ENGINE_Init fail 0x%x\n", ret); return 1; }

    Encoder enc;
    DecoderLoop dec;
    if (enc.Init(enc_path) != 0) { printf("encoder init fail\n"); return 1; }
    if (dec.Init(dec_path) != 0) { printf("decoder init fail\n"); return 1; }

    std::vector<float> feats;
    std::vector<float> pe;
    if (!read_file((in_dir + "/encoder_input.bin").c_str(), feats)) { printf("read feats fail\n"); return 1; }
    if (!read_file((in_dir + "/pe.bin").c_str(), pe)) { printf("read pe fail\n"); return 1; }

    int32_t length = read_int32_file((in_dir + "/encoder_input_lengths.bin").c_str());
    printf("input length %d\n", length);
    int T = feats.size() / 80;               // 1 x T x 80
    int max_feat = enc.GetInputSize(0) / 4 / 80;
    std::vector<float> feats_pad(max_feat * 80, 0.0f);
    memcpy(feats_pad.data(), feats.data(), feats.size() * sizeof(float));

    // encoder
    double t0 = now_ms();
    enc.SetInput(feats_pad.data(), 0);
    enc.SetInput(&length, 1);
    enc.Run();
    double t1 = now_ms();
    printf("encoder_ms %.3f\n", t1 - t0);

    int cross_k_n = enc.GetOutputSize(0) / 4;   // 16*1*250*1280
    int cross_v_n = enc.GetOutputSize(1) / 4;
    int mask_n = enc.GetOutputSize(2) / 4;
    std::vector<float> cross_k(cross_k_n), cross_v(cross_v_n), cross_mask(mask_n);
    enc.GetOutput(cross_k.data(), 0);
    enc.GetOutput(cross_v.data(), 1);
    enc.GetOutput(cross_mask.data(), 2);
    printf("cross_k[0..4]=%.4f %.4f %.4f %.4f %.4f\n",
           cross_k[0], cross_k[1], cross_k[2], cross_k[3], cross_k[4]);
    printf("cross_mask[0..4]=%.4f %.4f %.4f %.4f %.4f\n",
           cross_mask[0], cross_mask[1], cross_mask[2], cross_mask[3], cross_mask[4]);

    // decoder loop
    int Ti = cross_k_n / (16 * 1280);
    int hidden = 1280;
    int cache_len = dec.GetInputSize(1) / 4 / 16 / hidden;
    int n_layers = dec.GetInputSize(1) / 4 / cache_len / hidden;
    int vocab = dec.GetOutputSize(0) / 4;
    std::vector<float> k_cache(n_layers * cache_len * hidden, 0.0f);
    std::vector<float> v_cache(n_layers * cache_len * hidden, 0.0f);
    std::vector<int32_t> tokens = {3};
    std::vector<float> self_mask(cache_len, 0.0f);
    std::vector<float> pe_step(hidden);
    std::vector<float> logits(vocab);

    // 常量输入只设一次：cross_k/cross_v/cross_mask 全程不变
    dec.SetInput(cross_k.data(), 3);
    dec.SetInput(cross_v.data(), 4);
    dec.SetInput(cross_mask.data(), 7);
    // 初始 cache 全零
    dec.SetInput(k_cache.data(), 1);
    dec.SetInput(v_cache.data(), 2);

    double dec_total = 0.0;
    int steps = 0;
    for (int off = 0; off < max_steps; off++) {
        // self_attn_mask：前 cache_len-off-1 个位置 -inf（与 Python 基准一致）
        std::fill(self_mask.begin(), self_mask.end(), 0.0f);
        std::fill(self_mask.begin(), self_mask.begin() + (cache_len - off - 1), -INFINITY);
        memcpy(pe_step.data(), pe.data() + (size_t)off * hidden, hidden * sizeof(float));

        double s = now_ms();
        dec.SetInput(tokens.data(), 0);
        dec.SetInput(pe_step.data(), 5);
        dec.SetInput(self_mask.data(), 6);
        dec.Run();
        dec.GetOutput(logits.data(), 0);
        // 零拷贝：下一轮 cache 输入直接复用本轮输出 buffer
        dec.SetInputFromOutput(1, 1);
        dec.SetInputFromOutput(2, 2);
        double e = now_ms();
        dec_total += e - s;

        int best = 0;
        float bestv = logits[0];
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > bestv) { bestv = logits[i]; best = i; }
        }
        if (steps == 0) {
            std::vector<int> top5(5);
            std::vector<float> vals(5, -1e30f);
            for (int i = 0; i < vocab; i++) {
                for (int j = 0; j < 5; j++) {
                    if (logits[i] > vals[j]) {
                        for (int k = 4; k > j; k--) { vals[k] = vals[k-1]; top5[k] = top5[k-1]; }
                        vals[j] = logits[i]; top5[j] = i;
                        break;
                    }
                }
            }
            printf("step0 top5: %d %d %d %d %d\n", top5[0], top5[1], top5[2], top5[3], top5[4]);
        }
        tokens[0] = best;
        steps++;
        printf("%d ", best);
        if (best == 4) break;
    }
    printf("\ndecoder_steps %d\ndecoder_total_ms %.3f\ndecoder_per_step_ms %.3f\n",
           steps, dec_total, dec_total / steps);
    printf("whole_asr_ms %.3f\n", (now_ms() - t0));
    return 0;
}
