# FireRedASR.axera
FireRedASR-AED on Axera

小红书ASR AED-L版本在AX650N上的部署，原项目地址为：[https://github.com/FireRedTeam/FireRedASR](https://github.com/FireRedTeam/FireRedASR)

转换后的模型放置在axmodel目录，目前支持中文（部分方言）、英文。

[HuggingFace](https://huggingface.co/AXERA-TECH/FireRedASR-AED)上已有转换好的模型，最长支持10s输入，如需修改输入时长或最大token数目可使用本repo自行转换。

## 模型转换

[参考](model_convert/README.md)

## 支持平台

- [x] AX650N


## 安装依赖

### Python

测试环境为Python 3.12，建议使用[Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
)，安装方法[参考](https://www.anaconda.com/docs/getting-started/miniconda/install#aws-graviton2%2Farm64)

```
conda create -n fireredasr python=3.12
conda activate fireredasr
pip install -r requirements.txt
```

### 安装pyaxengine

```
wget https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc1/axengine-0.1.3-py3-none-any.whl
pip install axengine-0.1.3-py3-none-any.whl
```


## 使用

```
conda activate fireredasr
python test_ax_model.py
```