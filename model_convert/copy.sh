#!/bin/bash

mkdir -p ../axmodel
cp pretrained_models/FireRedASR-AED-L/dict.txt ../axmodel
cp pretrained_models/FireRedASR-AED-L/train_bpe1000.model ../axmodel
cp pretrained_models/FireRedASR-AED-L/cmvn.ark ../axmodel
cp onnx_decoder/pe.npy ../axmodel
cp axmodel_encoder/encoder.axmodel ../axmodel
cp axmodel_decoder_loop/decoder_loop.axmodel ../axmodel
