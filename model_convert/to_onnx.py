import model_wrapper
from fireredasr.models.fireredasr import FireRedAsrAed

import torch
import onnx
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic
import onnxslim
from onnx.external_data_helper import convert_model_to_external_data
import numpy as np
import math
import kaldiio

import os
import argparse
from typing import Dict, Any

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def load_model(model_path):
    package = torch.load(model_path, 
                         map_location=lambda storage, 
                         loc: storage, weights_only=False)
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model, package["args"]


def read_kaldi_cmvn(kaldi_cmvn_file):
    assert os.path.exists(kaldi_cmvn_file)
    stats = kaldiio.load_mat(kaldi_cmvn_file)
    assert stats.shape[0] == 2
    dim = stats.shape[-1] - 1
    count = stats[0, dim]
    assert count >= 1
    floor = 1e-20
    means = []
    inverse_std_variences = []
    for d in range(dim):
        mean = stats[0, d] / count
        means.append(mean.item())
        varience = (stats[1, d] / count) - mean*mean
        if varience < floor:
            varience = floor
        istd = 1.0 / math.sqrt(varience)
        inverse_std_variences.append(istd)
    return means, inverse_std_variences


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def calc_feat_len(audio_dur):
    import math
    sample_rate = 16000
    frame_length = 25 * sample_rate / 1000
    frame_shift = 10 * sample_rate / 1000
    length = math.floor((audio_dur * sample_rate - frame_length) / frame_shift) + 1
    return length


def export_encoder(fireredasr_model, args, model_args):
    encoder = model_wrapper.AudioEncoderTensorCache(
        fireredasr_model.encoder, 
        fireredasr_model.decoder)
    encoder.eval()

    # forge encoder input
    encoder_input = torch.randn(1, calc_feat_len(args.max_dur), 80)
    encoder_input_lengths = torch.tensor([100], dtype=torch.int64)
    
    n_layer_cross_k, n_layer_cross_v, cross_attn_mask = encoder(
        encoder_input, 
        encoder_input_lengths,
    )

    if not os.path.exists(args.encoder):
        os.makedirs(args.encoder, exist_ok=True)
    onnx_encoder_file = os.path.join(args.encoder, "encoder.onnx")

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (encoder_input, encoder_input_lengths),
            onnx_encoder_file,
            export_params=True,
            do_constant_folding=True,
            opset_version=16,
            verbose=False,
            input_names=["encoder_input",
                        "encoder_input_lengths"],
            output_names=["n_layer_cross_k", 
                        "n_layer_cross_v",
                        "cross_attn_mask"],
            external_data=True
        )
    
    external_filename = os.path.basename(onnx_encoder_file).split(".onnx")[0]
    model = onnx.load(onnx_encoder_file)
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,  
        location=f"./{external_filename}.data",          
        size_threshold=0,              
        convert_attribute=False        
    )

    onnx.save_model(
        model,
        onnx_encoder_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"./{external_filename}.data",
        size_threshold=0
    )

    onnx.checker.check_model(onnx_encoder_file, True)
    ort_session = onnxruntime.InferenceSession(onnx_encoder_file)
    onnx_encoder_input = to_numpy(encoder_input)
    onxx_encoder_input_lengths = to_numpy(encoder_input_lengths)
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_encoder_input,
                  ort_session.get_inputs()[1].name: onxx_encoder_input_lengths}
    ort_outputs = ort_session.run(None, ort_inputs)

    try:
        np.testing.assert_allclose(to_numpy(n_layer_cross_k), ort_outputs[0], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)
    try:
        np.testing.assert_allclose(to_numpy(n_layer_cross_v), ort_outputs[1], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)
    try:
        np.testing.assert_allclose(to_numpy(cross_attn_mask), ort_outputs[2], rtol=1e-03, atol=1e-05)
    except AssertionError as e:
        print(e)
        
    print("export onnx encoder done.")
    
    return n_layer_cross_k, n_layer_cross_v, cross_attn_mask


def export_decoder(fireredasr_model, args,
                   n_layer_cross_k,
                   n_layer_cross_v,
                   cross_attn_mask):
    beam_size = args.beam_size
    
    decoder = model_wrapper.TextDecoderTensorCache(
        fireredasr_model.decoder)
    decoder.eval()
    
    num_layer, batch_size, Ti, encoder_out_dim = n_layer_cross_k.shape
    encoder_out_length = cross_attn_mask.size(-1)
    
    # preparing for batch beam search
    cross_attn_mask = cross_attn_mask.unsqueeze(1).repeat(
        1, beam_size, 1, 1).view(beam_size * batch_size, -1, encoder_out_length)
    n_layer_cross_k = n_layer_cross_k.unsqueeze(2).repeat(
        1, 1, beam_size, 1, 1
    ).view(num_layer, beam_size * batch_size, Ti, encoder_out_dim)
    n_layer_cross_v = n_layer_cross_v.unsqueeze(2).repeat(
        1, 1, beam_size, 1, 1
    ).view(num_layer, beam_size * batch_size, Ti, encoder_out_dim)
    tokens = torch.ones(beam_size * batch_size, 1).fill_(decoder.decoder.sos_id).long()
    
    n_layer_self_k_cache = torch.zeros(
        (
            len(decoder.blocks),
            batch_size * beam_size,
            args.decode_max_len,
            1280
        )
    )
    n_layer_self_v_cache = torch.zeros(
        (
            len(decoder.blocks),
            batch_size * beam_size,
            args.decode_max_len,
            1280
        )
    )

    # decoder loop
    self_attn_mask = torch.empty(batch_size * beam_size, 
                                 1, args.decode_max_len
                                 ).fill_(-np.inf).triu_(1) # fill_(-np.inf)
    self_attn_mask = self_attn_mask[:, -1:, :]

    decoder = model_wrapper.TextDecoderTensorCacheV2(
        fireredasr_model.decoder, loop=True)
    decoder.eval()

    pe = decoder.decoder.positional_encoding.pe[0]
    pe_file = os.path.join(args.decoder, "pe.npy")
    np.save(pe_file, pe.numpy())

    onnx_decoder_file = os.path.join(args.decoder, "decoder_loop.onnx")

    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            pe[0],
            self_attn_mask,
            cross_attn_mask),
            onnx_decoder_file,
            export_params=True,
            opset_version=13,
            verbose=False,
            input_names=["tokens",
                        "in_n_layer_self_k_cache",
                        "in_n_layer_self_v_cache",
                        "n_layer_cross_k",
                        "n_layer_cross_v",
                        "pe",
                        "self_attn_mask",
                        "cross_attn_mask"],
            output_names=["logits",
                        "out_n_layer_self_k_cache",
                        "out_n_layer_self_v_cache"],
            external_data=True
        )
    print(f"Export decoder_loop to {onnx_decoder_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="export FireRedASR-AED torch model to onnx")
    parser.add_argument(
        "--model", 
        type=str, 
        default="pretrained_models/FireRedASR-AED-L/model.pth.tar",
        help="Path to FireRedASR-AED torch model"
    )
    parser.add_argument(
        "--encoder", 
        type=str, 
        default="onnx_encoder",
        help="Dir to the exported onnx encoder"
    )
    parser.add_argument(
        "--decoder", 
        type=str, 
        default="onnx_decoder",
        help="Dir to the exported onnx decoder"
    )
    parser.add_argument(
        "--cmvn",
        type=str,
        default="pretrained_models/FireRedASR-AED-L/cmvn.ark",
        help="cmvn.ark file"
    )
    parser.add_argument("--beam_size", type=int, default=3, help="")
    parser.add_argument(
        "--decode_max_len",
        type=int,
        required=False,
        default=128,
        help="decode max len"
    )
    parser.add_argument(
        "--max_dur",
        type=int,
        required=False,
        default=10,
        help="max audio len"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fireredasr_model, model_args = load_model(args.model)
    n_layer_cross_k, n_layer_cross_v, cross_attn_mask = export_encoder(fireredasr_model, args, model_args)
    export_decoder(fireredasr_model, args, n_layer_cross_k, n_layer_cross_v, cross_attn_mask)
    

if __name__ == "__main__":
    main()