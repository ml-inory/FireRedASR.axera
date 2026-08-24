# FireRedASR.axera
FireRedASR-AED on Axera

小红书ASR AED-L版本在AX650N上的部署，原项目地址为：[https://github.com/FireRedTeam/FireRedASR](https://github.com/FireRedTeam/FireRedASR)

转换后的模型放置在axmodel目录，目前支持中文（部分方言）、英文。

[HuggingFace](https://huggingface.co/AXERA-TECH/FireRedASR-AED)上已有转换好的模型，最长支持10s输入，如需修改输入时长或最大token数目可使用本repo自行转换。

## 模型转换

[参考](model_convert/README.md)

## 支持平台

- [x] AX650N

## VAD 说明（2026-08-24 更新）

- VAD：FSMN-VAD（FunASR 16k 通用）NPU 模型 `fsmn_vad/fsmn_vad_10s_fp32.axmodel`
  + `fsmn_vad/am.mvn`。10s 静态输入，按 10s 块做语音门控：
  块内检出语音 ≥ 1000ms 才整块送 ASR，静音/噪声块返回空。
- Python/C++ SDK 均内置 FSMN-VAD；无 torch/onnxruntime 依赖。


## 安装依赖

### Audio backend

```
sudo apt install libsndfile1
```

### Python

测试环境为Python 3.12，建议使用[Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
)，安装方法[参考](https://www.anaconda.com/docs/getting-started/miniconda/install#aws-graviton2%2Farm64)

```
conda create -n fireredasr python=3.12
conda activate fireredasr
pip install -r requirements.txt
```

```
$ export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/:$PYTHONPATH
```

### 安装pyaxengine

```
wget https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc1/axengine-0.1.3-py3-none-any.whl
pip install axengine-0.1.3-py3-none-any.whl
```


## 使用

Python（板端，FSMN-VAD + ASR 全 NPU）：

```bash
conda activate fireredasr
python openai/openai_server.py \
  --encoder axmodel/encoder.axmodel \
  --decoder axmodel/decoder_loop.axmodel \
  --fsmn-vad fsmn_vad/fsmn_vad_10s_fp32.axmodel \
  --fsmn-cmvn fsmn_vad/am.mvn \
  --cmvn axmodel/cmvn.ark --dict axmodel/dict.txt \
  --pe axmodel/pe.npy --max-dur 10 --max-steps 128
```

单文件转写可调用 `openai/firered_asr.py` 的 `FireredASR`；
C++ 端到端 SDK 见 `cpp/README.md`。

## Pulsar2 7.0-lite 重转换 + C++ SDK（2026-08）

- 用 Pulsar2 7.0-lite 重新转换 encoder(U16)+decoder(U8 密集校准)，
  精度与 HF 5.1 参考持平（CER 3.37%），全链路 RTF 0.225（C++ SDK）；
- `cpp/`：端到端 C++ SDK（板端 VAD+fbank+ASR，无 torch/onnxruntime）；
- `pulsar2_configs/`：可复现编译配置；
- `reports/`：对比报告与加速方案；
- `fsmn_vad/`：FSMN-VAD NPU 模型 + 编译配置 + 校准脚本；
- 详见 [cpp/README.md](cpp/README.md) 与 [reports/](reports/)。
