# FireRedASR-AED AX650N 全链路加速方案

基线（AX650N，Pulsar2 7.0-lite，U16 encoder + U8 dense decoder，
贪心解码，Python+numpy+pyaxengine 运行时）：

| 指标 | 数值 |
|------|------|
| CER（4 条测试语音） | 3.37% |
| 全链路 RTF | 0.318 |
| encoder（10s 输入） | 172ms |
| decoder NPU 单步（ax_run_model） | 23.1ms |
| decoder Python 实际单步 | ~53ms（含 ~30ms Python/pyaxengine 开销） |

## 瓶颈分解

- decoder 是绝对大头：5 段共 81 步，NPU 理论 1.87s，Python 实测 4.34s，
  每步额外开销约 30ms。
- encoder 其次：每段 ~191ms，且 U8 量化在 7.0 下不可用（cos≈0.11），
  只能 U16。
- 编译侧已探明：transformer_opt_level=2、tile、data soft compression、
  S8/MSE/Brecq(超时) 均无收益或掉精度；U16 decoder 有 cache 回写缺陷。

## 加速手段（按收益排序）

### 1. C++ 运行时（最大收益，已落地：RTF 0.318 → 0.211）

基于 HF 仓库 `EngineWrapper` 实现了 C++ 基准（`cpp/`）：
- decoder cache **零拷贝**（下一轮输入直接指向本轮输出 buffer），
  每步 20MB memcpy 消除；
- 常量输入（cross_k/v/mask）只设置一次；
- 实测 5 段测试集：decoder 每步 23.56ms（NPU 极限约 23.1ms），
  RTF 0.211（Python 优化版 0.318，提速约 34%），详见 `cpp/RESULTS.md`。

剩余：板端 fbank/cmvn、VAD、多段流水。

### 2. 缩短 encoder 输入时长 max_dur（10s → 5s）

已实测：encoder 减到 75.6ms/段，decoder 单步 ~20.8ms，
但 **5s decoder 的 U8 量化精度回退**（cos 0.987 vs 10s 的 0.9998，
贪心解码重复/错字）；U16 cache 回写缺陷依旧、BF16 编译不支持。
结论：**暂不建议采用**，需 QAT 或等工具链修复后再评估。

### 3. 缩短 decode_max_len（128 → 64）

- cache DDR 读写减半（每步 2×10MB → 2×5MB）；
- 自回归最坏步数减半，内存占用降低；
- 本测试集最长 38 步，64 足够。

### 4. 运行时流水线/并行

- 长音频多段：VAD/特征/encoder 与上一段 decoder 并行（C++ 多线程，
  Python 用 ThreadPool 也行，NPU 串行但 CPU 工作可重叠）；
- 板端 VAD 用 FSMN-VAD axmodel（`fsmn_vad/fsmn_vad_10s_fp32.axmodel`），避免主机回传。

### 5. 模型结构侧（中期，需重训/蒸馏）

- decoder 16 层 → 12 层（或 d_model 1280 → 1024）：decoder 每步约减
  25-35%，但需用原模型蒸馏/微调保精度；
- QAT：若要把 encoder 降到 INT8，7.0 的 PTQ 已证实失效，需
  QAT（QDQ 导出）才能拿到 172 → 115ms 的收益（encoder 占比小，收益有限）。

### 6. 已排除/不推荐

- transformer_opt_level=2：单步无提速（23.1ms），logits cos 0.9998→0.979；
- tile + data soft compression：23.5ms，无收益；
- S8/MSE/不同校准方法：更差；
- U16 decoder：cache 回写缺陷，不可用；
- U8 encoder：7.0 PTQ 下 cos≈0.11，不可用。

## 建议落地顺序

1. 先上 C++ 运行时（收益最大、不动模型）；
2. 再按产品约束决定是否把 max_dur 降到 5s / decode_max_len 降到 64
   （需重新导出+编译，精度风险低）；
3. 长音频加段间流水并行；
4. 若仍不够，再评估 decoder 蒸馏/QAT（周期长）。
