# FireRedASR-AED模型转换

以下步骤均在```model_convert```目录下进行

## 下载权重

```
cd pretrained_models
git lfs install
git clone https://huggingface.co/FireRedTeam/FireRedASR-AED-L
```

## 安装依赖

(使用Conda)
```
conda create -n fireredasr python=3.12
conda activate fireredasr
pip install -r ../requirements.txt
pip install onnx onnxruntime
```

## Torch -> ONNX

```
conda activate fireredasr
python to_onnx.py
```

通过参数```--decode_max_len```和```--max_dur```可以调整最大token数量和输入时长

运行完成后生成onnx_encoder和onnx_decoder目录

## 生成数据

```
python generate_data.py --max_dur 输入时长
```

## ONNX -> axmodel

### 获取Pulsar2工具链

[参考](https://huggingface.co/AXERA-TECH/Pulsar2)

### 转换axmodel

#### encoder
```
pulsar2 build --input onnx_encoder/encoder.onnx --config encoder.json --output_dir axmodel_encoder --output_name encoder.axmodel
```

#### decoder_main
```
pulsar2 build --input onnx_decoder/decoder_main.onnx --config decoder_main.json --output_dir axmodel_decoder_main --output_name decoder_main.axmodel
```

#### decoder_loop
```
pulsar2 build --input onnx_decoder/decoder_loop.onnx --config decoder_loop.json --output_dir axmodel_decoder_loop --output_name decoder_loop.axmodel
```

转换完成后运行copy.sh脚本
```
bash copy.sh
```

在../axmodel中包含了运行模型需要的文件