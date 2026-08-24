// FireRedASR-AED OpenAI 兼容客户端（CLI）
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "httplib.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: %s <server_url> <wav> [model]\n", argv[0]);
        printf("  example: %s http://10.126.35.166:8000 input.wav fireredasr-aed\n", argv[0]);
        return 1;
    }
    std::string url = argv[1];
    std::string wav = argv[2];
    std::string model = argc > 3 ? argv[3] : "fireredasr-aed";

    std::ifstream in(wav, std::ios::binary);
    if (!in) { printf("cannot open %s\n", wav.c_str()); return 1; }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string content = ss.str();

    httplib::Client cli(url);
    httplib::UploadFormDataItems items = {
        {"model", model, "", "text/plain"},
        {"file", content, wav, "audio/wav"},
    };
    auto res = cli.Post("/v1/audio/transcriptions", items);
    if (!res || res->status != 200) {
        printf("request failed: %s\n", res ? res->body.c_str() : "no response");
        return 1;
    }
    printf("%s\n", res->body.c_str());
    return 0;
}
