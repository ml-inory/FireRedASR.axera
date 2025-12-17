from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer

import axengine as axe
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple, List, Dict
import os
import time

INF = 1e10

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()
    
    
def set_finished_beam_score_to_zero(scores, is_finished):
    NB, B = scores.size()
    is_finished = is_finished.float()
    mask_score = torch.tensor([0.0] + [-INF]*(B-1)).float()
    mask_score = mask_score.view(1, B).repeat(NB, 1)
    return scores * (1 - is_finished) + mask_score * is_finished


def set_finished_beam_y_to_eos(ys, is_finished, eos_id):
    is_finished = is_finished.long()
    return ys * (1 - is_finished) + eos_id * is_finished


class FireRedASRAxModel:
    def __init__(
        self, 
        encoder_path: str, 
        decoder_loop_path: str,
        cmvn_file: str,
        dict_file: str, 
        spm_model_path: str,
        providers=['AxEngineExecutionProvider'],
        decode_max_len=128,
        audio_dur=10
    ):
        # NOTE: 参考whisper设置的最大的解码长度
        # FireRedASR-AED 模型支持的最长语音为 60s
        # ref: https://github.com/FireRedTeam/FireRedASR?tab=readme-ov-file#input-length-limitations
        self.decode_max_len = decode_max_len
        
        self.decoder_hidden_dim = 1280
        self.audio_dur = audio_dur
        self.max_feat_len = self.calc_feat_len(audio_dur)
        self.num_decoder_blocks = 16
        self.blank_id = 0
        self.sos_id = 3
        self.eos_id = 4
        self.pad_id = 2
        
        self.feature_extractor = ASRFeatExtractor(cmvn_file)
        self.tokenizer = ChineseCharEnglishSpmTokenizer(dict_file, spm_model_path)
        
        self.init_encoder(encoder_path, providers)
        self.init_decoder_loop(decoder_loop_path, providers)
        self.pe = self.init_pe(decoder_loop_path)
        
    def init_encoder(self, encoder_path, providers=None):
        self.encoder = axe.InferenceSession(
            encoder_path,
            providers=providers
        )

    def init_decoder_loop(self, decoder_path, providers=None):
        self.decoder_loop = axe.InferenceSession(
            decoder_path,
            providers=providers
        )

    def init_pe(self, decoder_path):
        decoder_path = os.path.dirname(decoder_path)
        decoder_path = os.path.join(decoder_path, "pe.npy")
        
        return np.load(decoder_path)
    
    def run_encoder(self, input: np.ndarray, 
                    input_length: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n_layer_cross_k, n_layer_cross_v, cross_attn_mask = self.encoder.run(
            None,
            {
                "encoder_input": input,
                "encoder_input_lengths": input_length
            }
        )
        return (
            n_layer_cross_k,
            n_layer_cross_v,
            cross_attn_mask
        )

    def decode_loop_one_token(
        self,
        tokens: np.ndarray,
        n_layer_self_k_cache: np.ndarray,
        n_layer_self_v_cache: np.ndarray,
        n_layer_cross_k_cache: np.ndarray,
        n_layer_cross_v_cache: np.ndarray,
        pe: np.ndarray,
        self_attn_mask: np.ndarray,
        cross_attn_mask: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = self.decoder_loop.run(
            None,
            {
                "tokens": tokens,
                "in_n_layer_self_k_cache": n_layer_self_k_cache,
                "in_n_layer_self_v_cache": n_layer_self_v_cache,
                "n_layer_cross_k": n_layer_cross_k_cache,
                "n_layer_cross_v": n_layer_cross_v_cache,
                "pe": pe,
                "self_attn_mask": self_attn_mask,
                "cross_attn_mask": cross_attn_mask,
            }
        )
        return (
            logits,
            out_n_layer_self_k_cache,
            out_n_layer_self_v_cache
        )
    
    def run_decoder(
        self,
        n_layer_cross_k, 
        n_layer_cross_v, 
        cross_attn_mask,
        beam_size,
        nbest
    ):
        
        num_layer, batch_size, Ti, encoder_out_dim = n_layer_cross_k.shape
        encoder_out_length = cross_attn_mask.shape[-1]
        
        cross_attn_mask = torch.from_numpy(cross_attn_mask).to(torch.float32)
        cross_attn_mask = cross_attn_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1
        ).view(beam_size * batch_size, -1, encoder_out_length)
        
        n_layer_cross_k = torch.from_numpy(n_layer_cross_k)
        n_layer_cross_v = torch.from_numpy(n_layer_cross_v)
        n_layer_cross_k = n_layer_cross_k.unsqueeze(2).repeat(
            1, 1, beam_size, 1, 1
        ).view(num_layer, beam_size * batch_size, Ti, encoder_out_dim)
        n_layer_cross_v = n_layer_cross_v.unsqueeze(2).repeat(
            1, 1, beam_size, 1, 1
        ).view(num_layer, beam_size * batch_size, Ti, encoder_out_dim)

        prediction_tokens = torch.ones(
            beam_size * batch_size, 1).fill_(self.sos_id).long()
        tokens = prediction_tokens
        offset = torch.zeros(1, dtype=torch.int64)
        n_layer_self_k_cache, n_layer_self_v_cache = self.get_initialized_self_cache(
            batch_size, beam_size
        )
        
        scores = torch.tensor([0.0] + [-INF]*(beam_size - 1)).float()
        scores = scores.repeat(batch_size).view(batch_size * beam_size, 1)
        is_finished = torch.zeros_like(scores)
        
        self_attn_mask = np.zeros((batch_size * beam_size, 1, 1), dtype=np.float32)

        for i in range(self.decode_max_len):

            tokens = to_numpy(tokens).astype(np.int32)
            n_layer_self_k_cache = to_numpy(n_layer_self_k_cache)
            n_layer_self_v_cache = to_numpy(n_layer_self_v_cache)
            n_layer_cross_k = to_numpy(n_layer_cross_k)
            n_layer_cross_v = to_numpy(n_layer_cross_v)
            cross_attn_mask = to_numpy(cross_attn_mask)

            self_attn_mask = np.zeros((batch_size * beam_size, 1, self.decode_max_len), dtype=np.float32)
            self_attn_mask[:, :, :self.decode_max_len - offset[0] - 1] = -np.inf

            logits, n_layer_self_k_cache, n_layer_self_v_cache = self.decode_loop_one_token(
                    to_numpy(tokens),
                    to_numpy(n_layer_self_k_cache),
                    to_numpy(n_layer_self_v_cache),
                    to_numpy(n_layer_cross_k),
                    to_numpy(n_layer_cross_v),
                    self.pe[offset],
                    self_attn_mask,
                    to_numpy(cross_attn_mask)
                )
            
            offset += 1
            logits = torch.from_numpy(logits)
            
            logits = logits.squeeze(1)
            t_scores = F.log_softmax(logits, dim=-1)
            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=beam_size, dim=1)
            t_topB_scores = set_finished_beam_score_to_zero(t_topB_scores, is_finished)
            t_topB_ys = set_finished_beam_y_to_eos(t_topB_ys, is_finished, self.eos_id)
            
            scores = scores + t_topB_scores
            
            scores = scores.view(batch_size, beam_size * beam_size)
            scores, topB_score_ids = torch.topk(scores, k=beam_size, dim=1)
            scores = scores.view(-1, 1)
            
            topB_row_number_in_each_B_rows_of_ys = torch.div(
                topB_score_ids, beam_size).view(batch_size * beam_size)
            stride = beam_size * torch.arange(batch_size).view(
                batch_size, 1).repeat(1, beam_size).view(batch_size * beam_size)
            topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()
            
            prediction_tokens = prediction_tokens[topB_row_number_in_ys]
            t_ys = torch.gather(
                t_topB_ys.view(batch_size, beam_size * beam_size), 
                dim=1, index=topB_score_ids
            ).view(beam_size * batch_size, 1)
            
            tokens = t_ys
        
            prediction_tokens = torch.cat((prediction_tokens, t_ys), dim=1)
            
            n_layer_self_k_cache = torch.from_numpy(n_layer_self_k_cache)
            n_layer_self_v_cache = torch.from_numpy(n_layer_self_v_cache)
            
            for i, self_k_cache in enumerate(n_layer_self_k_cache):
                n_layer_self_k_cache[i] = n_layer_self_k_cache[i][topB_row_number_in_ys]
            
            for i, self_v_cache in enumerate(n_layer_self_v_cache):
                n_layer_self_v_cache[i] = n_layer_self_v_cache[i][topB_row_number_in_ys]
                
            is_finished = t_ys.eq(self.eos_id)
            if is_finished.sum().item() == beam_size * batch_size:
                break
            
        scores = scores.view(batch_size, beam_size)
        prediction_valid_token_lengths = torch.sum(
            torch.ne(
                prediction_tokens.view(batch_size, beam_size, -1), 
                self.eos_id),
            dim=-1
        ).int()
       
        nbest_scores, nbest_ids = torch.topk(scores, k=nbest, dim=1)
        index = nbest_ids + beam_size * torch.arange(batch_size).view(batch_size, 1).long()
        nbest_prediction_tokens = prediction_tokens.view(batch_size * beam_size, -1)[index.view(-1)]
        nbest_prediction_tokens = nbest_prediction_tokens.view(batch_size, nbest_ids.size(1), -1)
        nbest_prediction_valid_token_lengths = prediction_valid_token_lengths.view(
            batch_size * beam_size)[index.view(-1)].view(batch_size, -1)
        nbest_hyps: List[List[Dict[str, torch.Tensor]]] = []
        for i in range(batch_size):
            i_best_hyps: List[Dict[str, torch.Tensor]] = []
            for j, score in enumerate(nbest_scores[i]):
                hyp = {
                    "token_ids": nbest_prediction_tokens[i, j, 1:nbest_prediction_valid_token_lengths[i, j]],
                    "score": score
                }
                i_best_hyps.append(hyp)
            nbest_hyps.append(i_best_hyps)

        return nbest_hyps
        
    def get_initialized_self_cache(self, 
                                   batch_size, 
                                   beam_size
                                   ) -> Tuple[Tensor, Tensor]:
        n_layer_self_k_cache = torch.zeros(
            self.num_decoder_blocks,
            batch_size * beam_size,
            self.decode_max_len,
            self.decoder_hidden_dim,
        )
        n_layer_self_v_cache = torch.zeros(
            self.num_decoder_blocks,
            batch_size * beam_size,
            self.decode_max_len,
            self.decoder_hidden_dim,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache
    
    def calc_feat_len(self, audio_dur):
        import math
        sample_rate = 16000
        frame_length = 25 * sample_rate / 1000
        frame_shift = 10 * sample_rate / 1000
        length = math.floor((audio_dur * sample_rate - frame_length) / frame_shift) + 1
        return length
 
    def transcribe(self, 
                   batch_wav_path: List[str],
                   beam_size: int = 1,
                   nbest: int = 1
                ) -> List[Dict]:
        feats, lengths, wav_durations = self.feature_extractor(batch_wav_path)
        # print(f"feats.shape: {feats.shape}")
        if feats.shape[1] < self.max_feat_len:
            feats = np.concatenate([feats, np.zeros((1, self.max_feat_len - feats.shape[1], 80), dtype=np.float32)], axis=1)
        feats = feats[:, :self.max_feat_len, :]
        lengths = torch.minimum(lengths, torch.tensor(self.max_feat_len))

        feats = to_numpy(feats)
        lengths = to_numpy(lengths).astype(np.int32)

        start_time = time.time()
        n_layer_cross_k, n_layer_cross_v, cross_attn_mask = self.run_encoder(
            to_numpy(feats),
            to_numpy(lengths)
        )
        # print(f"run encoder take {(time.time() - start_time) * 1000}ms")
        nbest_hyps = self.run_decoder(n_layer_cross_k,
                                      n_layer_cross_v,
                                      cross_attn_mask,
                                      beam_size,
                                      nbest,
                                      )
        transcribe_durations = time.time() - start_time
        results: List[Dict] = []
        for wav, hyp in zip(batch_wav_path, nbest_hyps):
            hyp = hyp[0]
            hyp_ids = [int(id) for id in hyp["token_ids"].cpu()]
            score = hyp["score"].item()
            text = self.tokenizer.detokenize(hyp_ids)
            results.append(
                {
                    "wav": wav,
                    "text": text,
                    "score": score
                }
            )
            
        return results, wav_durations, transcribe_durations   