# FireRedASR-AED · Pulsar2 7.0-lite 重转换 + C++ SDK

AX650N 上的 7.0-lite 重转换模型与端到端 C++ SDK（板端 VAD + fbank + ASR，
无 torch/onnxruntime 依赖）。

## 模型

- `axmodel/encoder.axmodel`（encoder U16）
- `axmodel/decoder_loop.axmodel`（decoder U8 密集校准）

## 指标（AX650N，4 条测试语音）

| 版本 | CER% | RTF |
|------|------|-----|
| Python 优化版 | 3.37 | 0.318 |
| C++ SDK | 3.37 | **0.225** |

## OpenAI API

Python 与 C++ 均提供 OpenAI 兼容 server/client：
- C++ 预编译：`cpp/bin/firered_openai_server` / `firered_openai_client`
- Python：`openai/openai_server.py` / `openai/openai_client.py`

## 使用

C++ SDK 源码、构建与运行说明、Pulsar2 编译配置见
[GitHub](https://github.com/ml-inory/FireRedASR.axera)（`cpp/`、
`pulsar2_configs/`、`reports/`）。
