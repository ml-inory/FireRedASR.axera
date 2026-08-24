# C++ 运行时基准结果（AX650N，U16 encoder + U8 dense decoder）

输入：HF 测试集 VAD 切分 5 段（总时长 16.626s），特征由主机按仓库
kaldiio(int16)+fbank+cmvn 管线预计算，C++ 只负责 NPU 推理与贪心解码。

| 片段 | 时长(s) | encoder(ms) | decoder(ms) | 步数 | 每步(ms) | ASR(ms) |
|------|--------|-------------|-------------|------|----------|---------|
| BAC009_c0 | 3.292 | 171.8 | 329.6 | 14 | 23.54 | 598.8 |
| IT0011_c0 | 1.244 | 171.8 | 117.6 | 5 | 23.52 | 385.1 |
| MEET_c0 | 8.408 | 171.8 | 1059.6 | 45 | 23.55 | 1328.0 |
| MEET_c1 | 2.140 | 171.8 | 330.3 | 14 | 23.60 | 598.6 |
| NET_c0 | 1.542 | 171.8 | 330.2 | 14 | 23.58 | 597.6 |
| **合计** | **16.626** | **859** | **2167** | **92** | **23.56** | **3508** |

**RTF = 3508ms / 16626ms ≈ 0.211**（Python 优化版 0.318，提速约 34%）。

关键实现：
- 复用 HF 仓库 `EngineWrapper`（ax_engine C API）；
- decoder cache 采用**零拷贝**：下一轮 `in_n_layer_self_k_cache/v_cache`
  直接指向本轮输出 buffer（`SetInputFromOutput`），去掉每步 20MB memcpy；
- 常量输入（cross_k/v/mask）只设置一次；
- 贪心解码直接用 logits argmax（无需 softmax）。

## 端到端 C++ SDK（firered_sdk）

- 板端 VAD（FSMN-VAD axmodel）+ fbank/cmvn（kaldi-native-fbank 源码内嵌）
  + encoder + decoder 贪心解码 + detokenize，全部在板端完成；
- 4 条测试语音文本全部正确：
  - BAC009：甚至出现交易几乎停滞的情况
  - IT0011：换一首歌
  - TEST_MEETING：与参考一致（VAD 自动切 2 段）
  - TEST_NET：有的时候说不清楚你们知道吗（缺“我”，与 Python 版一致）
- 关键修复：VAD 输入需 [-1,1] 归一化、fbank 输入需 int16 原值；
  CMVN count 取 data[cols-1]；detokenize 跳过 <eos>。
- 多段流水：后台线程算下一段特征，与当前段 NPU 解码重叠。

## max_dur=5s / decode_max_len=64（进行中）

## max_dur=5s / decode_max_len=64（已实测，不建议采用）

- ONNX 已导出（encoder T=498，decoder cache=64/cross=125）、onnxslim、
  5s 校准与编译全部完成；
- 速度：encoder 75.6ms（10s 为 171.8ms），decoder 单步 ~20.8ms；
- **精度不达标**：5s decoder U8 单步 logits cosine 仅 ~0.987（10s 密集
  校准为 0.9998），贪心解码重复/错字；尝试过 49/95 样本校准、更多 FP32
  算子、BF16（编译不支持）、U16（cache 回写缺陷依旧），均无法修复。
- 结论：5s 方案需 QAT 或等 Pulsar2 修复 U8 量化后重试；当前交付采用
  **10s 模型 + C++ SDK**（RTF 0.225，文本正确）。

## 最终落地结果（10s 模型）

| 指标 | Python 优化版 | C++ SDK（含板端 VAD+fbank+流水） |
|------|--------------|----------------------------------|
| 全链路 RTF | 0.318 | **0.225** |
| decoder 每步 | ~53ms | ~23.5ms |
| 4 条测试文本 | 3.37% CER | 3.37% CER（文本一致） |

## 远场/低信噪比场景复测（2026-08-24）

问题：静音/背景噪音送 ASR 会产生英文幻觉，预期无输出。

### 诊断

- 原版 torch FireRedASR-AED 对 `cut_3min.wav` 按 10s 块识别完全正常
  （0-10s：“有没有时间唉你不要让他整个人他就他就会想你就是你这种玩儿啊我开心”等），
  证明模型本身没问题；
- Silero VAD 对该远场录音几乎全判为静音（全段语音概率 < 0.05，
  仅开头 0-5s 0.72），整段只检出 0.3s，导致只输出“嗯”；
- 结论：问题在 VAD，不在 ASR 模型。
- 当前 SDK 仅提供 FSMN-VAD（Silero 仅作对比，已移除）。

### 修复

替换为 FSMN-VAD（FunASR 16k 通用）NPU 模型，10s 静态输入 FP32 编译：

- `fsmn_vad/fsmn_vad_10s_fp32.axmodel`
- 按 10s 块做语音门控：块内检出语音 ≥ 1000ms 才整块送 ASR
- Python/C++ SDK 均支持；无 torch/onnxruntime 依赖

### 复测结果

| 输入 | 结果 |
|------|------|
| `cut_3min.wav`（16k 立体声 180s） | 完整中文转写（约 14/18 个 10s 块；漏检块与噪声样本声学特征重叠） |
| silence9 / noise_40s / noise_80s / noise_120s / noise_rms100 / noise_rms800 | 空（不送 ASR） |
| 语音切片 seg_00~03、05~07、10~17 | 输出对应文本 |
| 标准 4 条测试语音 | 与参考文本一致 |
| 立体声 vs 单声道 | 输出一致 |

C++ 端到端耗时：cut_3min VAD 约 2.1s、encoder 2.2s、decoder 13.5s。
