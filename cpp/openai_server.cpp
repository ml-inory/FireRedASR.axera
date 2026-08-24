// FireRedASR-AED OpenAI 兼容 API 服务（AX650N）
#include <cstdio>
#include <cstring>
#include <string>

#include "httplib.h"
#include "sdk_lib.hpp"

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else o += c;
    }
    return o;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 10) {
        printf("usage: %s <encoder> <decoder> <fsmn_vad> <fsmn_cmvn> "
               "<cmvn> <dict> <pe.bin> <max_dur_s> <max_steps> "
               "[port] [vad_min_speech_ms]\n", argv[0]);
        return 1;
    }
    const char* enc = argv[1];
    const char* dec = argv[2];
    const char* fsmn_vad = argv[3];
    const char* fsmn_cmvn = argv[4];
    const char* cmvn = argv[5];
    const char* dict = argv[6];
    const char* pe = argv[7];
    int max_dur = atoi(argv[8]);
    int max_steps = atoi(argv[9]);
    int port = argc > 10 ? atoi(argv[10]) : 8000;
    int fsmn_min_ms = argc > 11 ? atoi(argv[11]) : 1000;

    FireredSdk sdk;
    if (!sdk.Init(enc, dec, fsmn_vad, fsmn_cmvn,
                  cmvn, dict, pe, max_dur, max_steps, fsmn_min_ms)) {
        printf("SDK init failed\n");
        return 1;
    }

    httplib::Server svr;
    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            "{\"object\":\"list\",\"data\":[{\"id\":\"fireredasr-aed\","
            "\"object\":\"model\",\"owned_by\":\"axera\"}]}",
            "application/json");
    });
    svr.Post("/v1/audio/transcriptions", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.form.has_file("file")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing multipart field 'file'\"}", "application/json");
            return;
        }
        auto f = req.form.get_file("file");
        const std::string tmp = "/tmp/firered_openai_input.wav";
        FILE* fp = fopen(tmp.c_str(), "wb");
        if (!fp) {
            res.status = 500;
            res.set_content("{\"error\":\"cannot write temp wav\"}", "application/json");
            return;
        }
        fwrite(f.content.data(), 1, f.content.size(), fp);
        fclose(fp);
        double vad_ms = 0, enc_ms = 0, dec_ms = 0;
        std::string text = sdk.Transcribe(tmp.c_str(), &vad_ms, &enc_ms, &dec_ms);
        std::string body = "{\"text\":\"" + json_escape(text) + "\"}";
        res.set_content(body, "application/json");
    });
    printf("OpenAI API listening on 0.0.0.0:%d\n", port);
    svr.listen("0.0.0.0", port);
    return 0;
}
