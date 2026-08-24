# FireRedASR-AED AX650N 重转换对比报告（Pulsar2 7.0-lite）

日期：2026-08-24

## 结论

用 Pulsar2 7.0-lite 重新转换 FireRedASR-AED（参考
`ml-inory/FireRedASR.axera`），在 AX650N 上与 HF 上
`AXERA-TECH/FireRedASR-AED`（实际由 Pulsar2 5.1 编译）对比：

- **精度持平**：最优 7.0 组合（encoder U16 + decoder U8/密集校准）
  CER 3.37%，HF 5.1 参考 CER 3.37%（4 条测试语音，VAD 切 5 段）。
- **速度基本持平**：全链路 RTF 0.329 vs 0.323；decoder 单步
  23.3ms vs 23.4ms；encoder 10s 输入 171.8ms vs 157.5ms。

## 关键发现

1. **HF 仓库本体并未损坏**：官方仓库 `AXERA-TECH/FireRedASR-AED`
   的 LFS sha256 为 `6cc674ba...`，与 ModelScope 下载的完好文件完全一致
   （板端可加载，cross cosine≈0.9997）。最初经 hf-mirror 下载的副本
   被镜像/CDN 损坏（33.5% 字节为 0、尾部约 50MB 全零，SHA
   `35cdb624...`），导致“HF encoder 损坏”的误判。改用 ModelScope 或
   官方 huggingface.co 下载即可。
2. **7.0-lite 的 U8 量化对校准数据敏感**：用稀疏校准（22 样本）时
   decoder 单步 logits cosine 仅 0.976，贪心解码重复/乱码（CER 85%）；
   改用密集校准（41 样本，解码步每 2 步采样）后 cosine 升至 0.9998，
   与 HF 5.1（0.9998）持平。
3. **7.0-lite 的 U16 decoder 存在 cache 回写缺陷**：单步 logits
   cosine 0.995，但多步解码时 cache 几乎不更新，持续重复首 token，
   不可用（U16 混合精度同样复现）。
4. **7.0-lite 的 U8 encoder 不可用**：cross_k/v cosine 仅 0.11/-0.002，
   即使配 HF decoder 也乱码；encoder 应使用 U16。

## 量化配置探索矩阵（decoder 单步 logits，同一组 ONNX 输入）

| 配置 | cosine | 单步耗时 | 备注 |
|------|--------|----------|------|
| HF 5.1（官方仓库配置） | 0.999846 | 23.35ms | 参考 |
| 7.0 U8 + 稀疏校准（仓库配置） | 0.976 | 23.23ms | 重复/乱码 |
| 7.0 U8 + 关键算子 FP32 | 0.976 | 23.65ms | 无改善 |
| 7.0 U8 + 关 SmoothQuant/开 auto-refine | 0.976 | 23.21ms | 无改善 |
| 7.0 U8 + 关量化优化 | 0.976 | 23.23ms | 无改善 |
| 7.0 U8 + NPU 后端精度分析 | 0.976 | 23.18ms | 无改善 |
| 7.0 U8 + MSE 校准 | 0.959 | 23.35ms | 更差 |
| 7.0 S8 | 0.946 | 23.78ms | 更差 |
| 7.0 U8 + 密集校准（推荐） | 0.999806 | 23.27ms | **与 5.1 持平** |
| 7.0 U16（全 U16） | 0.995 | 26.41ms | cache 回写缺陷 |
| 7.0 U16 混合（MatMul U16） | 0.995 | 26.81ms | cache 回写缺陷 |

## 全链路精度/速度（AX650N，VAD 切分 5 段，贪心解码）

| 组合 | CER% | avg RTF | encoder/段 | decoder/段 |
|------|------|---------|------------|------------|
| HF 5.1（官方 enc + HF dec） | 3.37 | 0.3234 | 178ms | 897ms |
| 7.0 U16 enc + HF 5.1 dec | 3.37 | 0.3342 | 195ms | 917ms |
| 7.0 U16 enc + 7.0 U8 稀疏 dec | 85.4 | 0.26 | — | — |
| 7.0 U16 enc + 7.0 U8 密集 dec（推荐） | 3.37 | 0.3291 | 195ms | 899ms |
| 7.0 U8 enc + HF 5.1 dec | 182 | 0.4835 | — | — |

## 推荐交付配置

- encoder：`compile/encoder_u16_ref.axmodel`（812MB，U16，cos 0.9997）
- decoder：`compile/decoder_loop_u8_dense.axmodel`（398MB，U8，密集校准，
  cos 0.9998）
- 校准数据：`export/calib_data/`（4 条真实语音，decoder 每 2 步采样）
- 板端运行：numpy + pyaxengine，不依赖 torch/onnxruntime

## 复现

- 导出：`origin/FireRedASR.axera/model_convert/to_onnx.py`
  （beam_size=1, decode_max_len=128, max_dur=10；导出后 onnxslim
  处理供 7.0 编译，或直接用原始 ONNX）
- 校准：`generate_data.py`（CALIB_EVERY=2, CALIB_MAX_STEPS=128）
  → `compile/make_calib_tars.py`
- 编译：`compile/pulsar2_decoder_u8_dense.json` /
  `compile/pulsar2_encoder_u16_ref_slim.json`（Pulsar2 7.0-lite，
  `~/.cache/magnetar/pulsar2/7.0`）
