# FireRedASR-AED C++ SDK（AX650N，Pulsar2 7.0-lite 重转换）

端到端板端 SDK：wav → VAD(FSMN-VAD axmodel) → fbank/cmvn(kaldi-native-fbank)
→ encoder → decoder 贪心解码 → 文本。只依赖 ax_engine + libstdc++，
不依赖 torch/onnxruntime。

## 构建

需要 AX650 BSP（`msp/out`：include/ax_engine_api.h + lib/libax_engine*）与
aarch64 交叉编译器（gcc-arm-9.2）。

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=<path>/aarch64-none-linux-gnu.toolchain.cmake
make -j8
```

CMake 变量：
- `BSP_MSP_DIR`：AX650 BSP 的 `msp/out` 目录
- `FIRERED_CPP`：本仓库 cpp/ 源码目录（含 EngineWrapper）

## 运行

```bash
./firered_sdk encoder.axmodel decoder_loop.axmodel \
  fsmn_vad/fsmn_vad_10s_fp32.axmodel fsmn_vad/am.mvn \
  input.wav cmvn.ark dict.txt 10 128 pe.bin 1000
```

输出 `text:` 为识别文本；`enc_ms/dec_ms/asr_ms` 为各阶段耗时。
最后一个参数是块级最小语音时长（默认 1000ms）。

## OpenAI API

- `firered_openai_server`：OpenAI 兼容服务（`/v1/models`、
  `/v1/audio/transcriptions`）：`./firered_openai_server <encoder> <decoder> <fsmn_vad> <fsmn_cmvn> <cmvn> <dict> <pe.bin> <max_dur> <max_steps> [port] [vad_min_speech_ms]`
- `firered_openai_client`：`./firered_openai_client http://<host>:8000 input.wav fireredasr-aed`
- Python 版 server/client 见仓库 `openai/`（stdlib，零额外依赖）

## 已知结论（详见 reports/）

- 10s 模型 + C++ SDK：RTF ≈ 0.225（Python 版 0.318），CER 3.37%
- 5s/decode64 模型速度更快但 7.0 U8 decoder 精度回退（cos 0.987），暂不采用
- decoder cache 使用零拷贝（输出 buffer 直接作为下一轮输入），
  每步从 ~53ms 降到 ~23.5ms
