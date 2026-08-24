// FireRedASR-AED AX650N 端到端 CLI：wav -> 文本
#include <cstdio>

#include "sdk_lib.hpp"

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 9) {
        printf("usage: %s <encoder.axmodel> <decoder_loop.axmodel> "
               "<fsmn_vad.axmodel> <fsmn_am.mvn> "
               "<wav> <cmvn.ark> <dict.txt> <max_dur_s> [max_steps] [pe.bin] "
               "[vad_min_speech_ms]\n", argv[0]);
        return 1;
    }
    const char* enc = argv[1];
    const char* dec = argv[2];
    const char* fsmn_vad = argv[3];
    const char* fsmn_cmvn = argv[4];
    const char* wav = argv[5];
    const char* cmvn = argv[6];
    const char* dict = argv[7];
    int max_dur = atoi(argv[8]);
    int max_steps = argc > 9 ? atoi(argv[9]) : 128;
    const char* pe = argc > 10 ? argv[10] : "/tmp/firered_sdk/pe.bin";
    int fsmn_min_ms = argc > 11 ? atoi(argv[11]) : 1000;

    FireredSdk sdk;
    if (!sdk.Init(enc, dec, fsmn_vad, fsmn_cmvn,
                  cmvn, dict, pe, max_dur, max_steps, fsmn_min_ms)) {
        printf("SDK init failed\n");
        return 1;
    }
    double vad_ms = 0, enc_ms = 0, dec_ms = 0;
    std::string text = sdk.Transcribe(wav, &vad_ms, &enc_ms, &dec_ms);
    printf("vad_ms %.1f enc_ms %.1f dec_ms %.1f asr_ms %.1f\n",
           vad_ms, enc_ms, dec_ms, vad_ms + enc_ms + dec_ms);
    printf("text: %s\n", text.c_str());
    return 0;
}
