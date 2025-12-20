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
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    torchaudio.set_audio_backend("soundfile")
except Exception as e:
    print("Please run apt install libsnffile1 first")
    raise e

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

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
    mask_score = torch.tensor([0.0] + [-INF] * (B - 1)).float()
    mask_score = mask_score.view(1, B).repeat(NB, 1)
    return scores * (1 - is_finished) + mask_score * is_finished


def set_finished_beam_y_to_eos(ys, is_finished, eos_id):
    is_finished = is_finished.long()
    return ys * (1 - is_finished) + eos_id * is_finished


def expand_for_beam_search(n_layer_cross_k, beam_size):
    """方法1: 使用expand_dims + tile + reshape (最快)"""
    num_layer, batch_size, Ti, encoder_out_dim = n_layer_cross_k.shape
    
    # 在第2维插入新维度
    expanded = np.expand_dims(n_layer_cross_k, axis=2)
    # 使用tile替代repeat，性能更好
    tiled = np.tile(expanded, (1, 1, beam_size, 1, 1))
    # 重塑形状
    reshaped = tiled.reshape(num_layer, beam_size * batch_size, Ti, encoder_out_dim)
    
    return reshaped


class FireRedASRAxModel:
    def __init__(self,
            encoder_path: str,
            decoder_loop_path: str,
            cmvn_file: str,
            dict_file: str,
            spm_model_path: str,
            providers=["AxEngineExecutionProvider"],
            decode_max_len=128,
            audio_dur=10):
        # NOTE: 参考whisper设置的最大的解码长度
        # FireRedASR-AED 模型支持的最长语音为 60s
        # ref: https://github.com/FireRedTeam/FireRedASR?tab=readme-ov-file#input-length-limitations
        self.decode_max_len = decode_max_len
        self.sample_rate = 16000
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

        self.vad_model = load_silero_vad()

        # 预分配内存
        self._preallocated_memory()
        # 启用CUDA如果可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {self.device}")

    def calc_feat_len(self, audio_dur):
        import math

        sample_rate = self.sample_rate
        frame_length = 25 * sample_rate / 1000
        frame_shift = 10 * sample_rate / 1000
        length = math.floor((audio_dur * sample_rate - frame_length) / frame_shift) + 1
        return length
    
    def init_encoder(self, encoder_path, providers=None):
        self.encoder = axe.InferenceSession(encoder_path, providers=providers)

    def init_decoder_loop(self, decoder_path, providers=None):
        self.decoder_loop = axe.InferenceSession(decoder_path, providers=providers)

    def init_pe(self, decoder_path):
        decoder_path = os.path.dirname(decoder_path)
        decoder_path = os.path.join(decoder_path, "pe.npy")

        return np.load(decoder_path)
    
    def run_encoder(
        self, input: np.ndarray, input_length: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n_layer_cross_k, n_layer_cross_v, cross_attn_mask = self.encoder.run(
            None, {"encoder_input": input, "encoder_input_lengths": input_length}
        )
        return (n_layer_cross_k, n_layer_cross_v, cross_attn_mask)
    
    def decode_loop_one_token(
        self,
        tokens: np.ndarray,
        n_layer_self_k_cache: np.ndarray,
        n_layer_self_v_cache: np.ndarray,
        n_layer_cross_k_cache: np.ndarray,
        n_layer_cross_v_cache: np.ndarray,
        pe: np.ndarray,
        self_attn_mask: np.ndarray,
        cross_attn_mask: np.ndarray,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        (
            logits,
            out_n_layer_self_k_cache,
            out_n_layer_self_v_cache,
        ) = self.decoder_loop.run(
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
            },
        )
        return (logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache)
        
    def _preallocated_memory(self):
        """预分配常用内存空间"""
        # 预计算self_attn_mask模板
        self.self_attn_mask_templates = {}
        for offset in range(self.decode_max_len):
            mask = np.zeros((1, 1, self.decode_max_len), dtype=np.float32)
            mask[:, :, :self.decode_max_len - offset - 1] = -np.inf
            self.self_attn_mask_templates[offset] = mask
        
        # 预分配beam search的scores模板
        self.beam_scores_template = torch.tensor(
            [0.0] + [-INF] * (self.decode_max_len - 1)
        ).float()
        
    def transcribe(
        self, 
        batch_wav_path: List[str], 
        beam_size: int = 1, 
        nbest: int = 1,
        use_parallel: bool = False
    ) -> List[Dict]:
        """优化后的转录方法"""
        
        # 1. 优化VAD和分块处理
        chunks = self._optimized_vad_split(batch_wav_path[0])
        
        if use_parallel and len(chunks) > 1:
            return self._parallel_transcribe(chunks, beam_size, nbest)
        else:
            return self._sequential_transcribe(chunks, beam_size, nbest)
    
    def _optimized_vad_split(self, wav_path: str) -> List[torch.Tensor]:
        """优化的VAD分块处理"""
        import torchaudio
        
        # 直接读取为numpy数组，避免torchaudio开销
        try:
            wav, sr = torchaudio.load(wav_path)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        except:
            # 使用silero_vad的read_audio作为备选
            from silero_vad import read_audio
            wav = read_audio(wav_path, sampling_rate=self.sample_rate)
            wav = wav.unsqueeze(0)
        
        wav = wav.squeeze(0)
        
        # 快速VAD：如果音频较短，直接返回
        max_chunk_samples = int(self.sample_rate * self.audio_dur)
        if wav.shape[0] < max_chunk_samples:
            return [wav]
        
        # 使用优化的VAD参数
        speech_timestamps = get_speech_timestamps(
            wav,
            self.vad_model,
            threshold=0.5,  # 提高阈值，减少静音检测
            min_speech_duration_ms=250,  # 最小语音段
            min_silence_duration_ms=100,  # 最小静音段
            return_seconds=False,
        )
        
        # 优化的分块合并算法
        return self._optimized_collect_chunks(wav, speech_timestamps)
    
    def _optimized_collect_chunks(
        self, 
        wav: torch.Tensor, 
        speech_timestamps: List[Dict]
    ) -> List[torch.Tensor]:
        """优化的分块合并算法"""
        max_chunk_samples = int(self.sample_rate * self.audio_dur)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for ts in speech_timestamps:
            start, end = ts["start"], ts["end"]
            chunk_length = end - start
            
            if current_length + chunk_length <= max_chunk_samples:
                current_chunk.append((start, end))
                current_length += chunk_length
            else:
                if current_chunk:
                    # 合并当前chunk
                    merged = torch.cat([wav[s:e] for s, e in current_chunk])
                    chunks.append(merged)
                
                if chunk_length > max_chunk_samples:
                    # 大chunk分割
                    num_splits = (chunk_length + max_chunk_samples - 1) // max_chunk_samples
                    for i in range(num_splits):
                        s = start + i * max_chunk_samples
                        e = min(start + (i + 1) * max_chunk_samples, end)
                        chunks.append(wav[s:e])
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [(start, end)]
                    current_length = chunk_length
        
        # 处理最后一个chunk
        if current_chunk:
            merged = torch.cat([wav[s:e] for s, e in current_chunk])
            chunks.append(merged)
        
        return chunks
    
    def _optimized_decode_loop(
        self, 
        n_layer_cross_k: np.ndarray,
        n_layer_cross_v: np.ndarray,
        cross_attn_mask: np.ndarray,
        beam_size: int,
        nbest: int
    ) -> List[Dict]:
        """优化的解码循环"""
        
        num_layer, batch_size, Ti, encoder_out_dim = n_layer_cross_k.shape
        encoder_out_length = cross_attn_mask.shape[-1]

        n_layer_cross_k = expand_for_beam_search(n_layer_cross_k, beam_size)
        n_layer_cross_v = expand_for_beam_search(n_layer_cross_v, beam_size)

        batch_size, Ti, encoder_out_length = cross_attn_mask.shape
    
        # 在第1维插入新维度
        expanded = np.expand_dims(cross_attn_mask, axis=1)
        # 使用tile替代repeat，性能更好
        tiled = np.tile(expanded, (1, beam_size, 1, 1))
        # 重塑形状
        cross_attn_mask = tiled.reshape(beam_size * batch_size, Ti, encoder_out_length)
            
        # 优化的cache初始化
        n_layer_self_k_cache, n_layer_self_v_cache = self._optimized_init_self_cache(
            batch_size, beam_size
        )
        
        # 预分配tokens和scores
        tokens = torch.full(
            (beam_size * batch_size, 1), 
            self.sos_id, 
            dtype=torch.int32, device=self.device
        )
        scores = self.beam_scores_template[:beam_size].repeat(batch_size).view(
            batch_size * beam_size, 1
        ).to(self.device)
        is_finished = torch.zeros_like(scores, dtype=torch.bool, device=self.device)
        
        # 预分配prediction_tokens
        prediction_tokens = tokens.clone()
        
        pe_np = self.pe
        
        for offset in range(self.decode_max_len):
            # 使用预计算的mask模板
            self_attn_mask = np.repeat(
                self.self_attn_mask_templates[offset], 
                beam_size * batch_size, 
                axis=0
            )
            
            # 直接使用numpy数组，避免转换
            logits, n_layer_self_k_cache, n_layer_self_v_cache = (
                self.decode_loop_one_token(
                    tokens.cpu().numpy().astype(np.int32),
                    n_layer_self_k_cache,
                    n_layer_self_v_cache,
                    n_layer_cross_k,
                    n_layer_cross_v,
                    pe_np[offset],
                    self_attn_mask,
                    cross_attn_mask
                )
            )
            
            logits = torch.from_numpy(logits).to(self.device).squeeze(1)
            t_scores = F.log_softmax(logits, dim=-1)
            
            # 优化的beam search
            tokens, scores, prediction_tokens, n_layer_self_k_cache, n_layer_self_v_cache, is_finished = (
                self._optimized_beam_search(
                    t_scores, tokens, scores, prediction_tokens,
                    n_layer_self_k_cache, n_layer_self_v_cache,
                    is_finished, beam_size, batch_size
                )
            )
            
            if is_finished.all():
                break
        
        # return self._extract_results(scores, prediction_tokens, batch_size, beam_size, nbest)
        return self.extract_results_numpy_vectorized(scores.numpy(), prediction_tokens.numpy(), batch_size, beam_size, nbest)
    
    
    def _optimized_beam_search(
        self,
        t_scores: torch.Tensor,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        prediction_tokens: torch.Tensor,
        n_layer_self_k_cache: torch.Tensor,
        n_layer_self_v_cache: torch.Tensor,
        is_finished: torch.Tensor,
        beam_size: int,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """优化的beam search步骤"""
        
        # 使用torch的in-place操作
        t_topB_scores, t_topB_ys = torch.topk(t_scores, k=beam_size, dim=1)
        
        # 处理已完成的beam
        if is_finished.any():
            # 原地操作，避免创建新tensor
            t_topB_scores.masked_fill_(is_finished, 0.0)
            t_topB_scores[:, 1:].masked_fill_(is_finished, -INF)
            t_topB_ys.masked_fill_(is_finished, self.eos_id)
        
        # 更新scores
        scores = scores + t_topB_scores
        
        # 优化的topk选择
        scores_2d = scores.view(batch_size, beam_size * beam_size)
        top_scores, top_ids = torch.topk(scores_2d, k=beam_size, dim=1)
        scores = top_scores.view(-1, 1)
        
        # 计算索引
        topB_row_number_in_each_B_rows_of_ys = torch.div(top_ids, beam_size, rounding_mode='floor')
        stride = beam_size * torch.arange(batch_size, device=self.device).view(batch_size, 1)
        topB_row_number_in_ys = (topB_row_number_in_each_B_rows_of_ys + stride).view(-1)
        
        # 更新tokens和prediction_tokens
        tokens = torch.gather(
            t_topB_ys.view(batch_size, beam_size * beam_size),
            dim=1,
            index=top_ids,
        ).view(beam_size * batch_size, 1)
        
        prediction_tokens = torch.cat([
            prediction_tokens[topB_row_number_in_ys],
            tokens
        ], dim=1)
        
        # 更新cache（原地操作）
        for i in range(n_layer_self_k_cache.shape[0]):
            n_layer_self_k_cache[i] = n_layer_self_k_cache[i][topB_row_number_in_ys]
            n_layer_self_v_cache[i] = n_layer_self_v_cache[i][topB_row_number_in_ys]
        
        # 更新完成状态
        is_finished = tokens.eq(self.eos_id)
        
        return tokens, scores, prediction_tokens, n_layer_self_k_cache, n_layer_self_v_cache, is_finished
    
    def _optimized_init_self_cache(
        self, batch_size: int, beam_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """优化的self cache初始化"""
        shape = (
            self.num_decoder_blocks,
            batch_size * beam_size,
            self.decode_max_len,
            self.decoder_hidden_dim
        )
        n_layer_self_k_cache = np.zeros(shape, dtype=np.float32)
        n_layer_self_v_cache = np.zeros(shape, dtype=np.float32)
        return n_layer_self_k_cache, n_layer_self_v_cache
    
    def _extract_results(
        self,
        scores: torch.Tensor,
        prediction_tokens: torch.Tensor,
        batch_size: int,
        beam_size: int,
        nbest: int
    ) -> List[Dict]:
        """提取结果"""
        scores = scores.view(batch_size, beam_size)
        valid_lengths = torch.sum(
            torch.ne(prediction_tokens.view(batch_size, beam_size, -1), self.eos_id),
            dim=-1
        ).int()
        
        nbest_scores, nbest_ids = torch.topk(scores, k=nbest, dim=1)
        index = nbest_ids + beam_size * torch.arange(batch_size, device=self.device).unsqueeze(1)
        
        nbest_tokens = prediction_tokens.view(batch_size * beam_size, -1)[index.view(-1)]
        nbest_tokens = nbest_tokens.view(batch_size, nbest_ids.size(1), -1)
        
        results = []
        for j, score in enumerate(nbest_scores[0]):
            hyp = {
                "token_ids": nbest_tokens[0, j, 1:valid_lengths[0, nbest_ids[0, j]]],
                "score": score,
            }
            results.append(hyp)
        
        return results
    
    
    def extract_results_numpy_vectorized(
        self,
        scores: np.ndarray,
        prediction_tokens: np.ndarray,
        batch_size: int,
        beam_size: int,
        nbest: int,
        eos_id: int = 4
    ) -> List[Dict]:
        """向量化版本的NumPy实现"""
        
        # 1. 重塑和计算有效长度
        scores_2d = scores.reshape(batch_size, beam_size)
        tokens_3d = prediction_tokens.reshape(batch_size, beam_size, -1)
        
        # 计算有效长度（不包括eos_id）
        valid_lengths = np.sum(tokens_3d != eos_id, axis=-1).astype(np.int32)
        
        # 2. 使用argpartition进行部分排序（比argsort更快）
        # 获取最大的nbest个元素的索引
        # 使用argpartition: O(n) vs argsort: O(n log n)
        partitioned_indices = np.argpartition(-scores_2d, nbest-1, axis=1)[:, :nbest]
        
        # 对每个batch内的topk进行排序
        nbest_scores = np.take_along_axis(scores_2d, partitioned_indices, axis=1)
        sorted_order = np.argsort(-nbest_scores, axis=1)
        
        # 应用排序
        nbest_ids = np.take_along_axis(partitioned_indices, sorted_order, axis=1)
        nbest_scores = np.take_along_axis(nbest_scores, sorted_order, axis=1)
        
        # 3. 计算全局索引
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        global_indices = nbest_ids + beam_size * batch_indices
        flat_global_indices = global_indices.reshape(-1)
        
        # 4. 提取tokens
        flat_tokens = prediction_tokens.reshape(-1, prediction_tokens.shape[-1])
        nbest_tokens = flat_tokens[flat_global_indices]
        nbest_tokens = nbest_tokens.reshape(batch_size, nbest, -1)
        
        # 5. 提取对应的有效长度
        nbest_valid_lengths = np.take_along_axis(valid_lengths, nbest_ids, axis=1)
        
        # 6. 构建结果
        results = []
        
        for b in range(batch_size):
            batch_results = []
            for j in range(nbest):
                valid_len = nbest_valid_lengths[b, j]
                
                # 提取token_ids（跳过<sos>）
                token_ids = nbest_tokens[b, j, 1:valid_len]
                
                hyp = {
                    "token_ids": token_ids.tolist(),
                    "score": float(nbest_scores[b, j]),
                }
                batch_results.append(hyp)
            
            # 如果是批量处理，可以按batch返回
            # 这里假设batch_size=1，直接返回第一个batch的结果
            if b == 0:
                results = batch_results
        
        return results
    

    def _sequential_transcribe(
        self, 
        chunks: List[torch.Tensor], 
        beam_size: int, 
        nbest: int
    ) -> Dict:
        """顺序转录（单线程）"""
        tokens = []
        wav_durations = []
        transcribe_duration = 0
        
        for chunk in chunks:
            # 优化的特征提取
            feats, lengths, wav_duration = self._optimized_feature_extraction(chunk)
            wav_durations.append(wav_duration)
            
            # 运行encoder和decoder
            start_time = time.time()
            n_layer_cross_k, n_layer_cross_v, cross_attn_mask = self.run_encoder(
                feats, lengths.numpy().astype(np.int32)
            )
            
            nbest_hyps = self._optimized_decode_loop(
                n_layer_cross_k, n_layer_cross_v, cross_attn_mask, beam_size, nbest
            )
            
            tokens.extend([int(id) for id in nbest_hyps[0]["token_ids"]])
            transcribe_duration += time.time() - start_time
        
        text = self.tokenizer.detokenize(tokens)
        return {"text": text}, wav_durations, transcribe_duration
    
    def _parallel_transcribe(
        self, 
        chunks: List[torch.Tensor], 
        beam_size: int, 
        nbest: int
    ) -> Dict:
        """并行转录（多线程）"""
        import threading
        
        results = []
        lock = threading.Lock()
        
        def process_chunk(chunk_idx, chunk):
            try:
                # 特征提取
                feats, lengths, wav_duration = self._optimized_feature_extraction(chunk)
                
                # encoder
                n_layer_cross_k, n_layer_cross_v, cross_attn_mask = self.run_encoder(
                    feats, lengths.astype(np.int32)
                )
                
                # decoder
                nbest_hyps = self._optimized_decode_loop(
                    n_layer_cross_k, n_layer_cross_v, cross_attn_mask, beam_size, nbest
                )
                
                with lock:
                    results.append({
                        'chunk_idx': chunk_idx,
                        'tokens': [int(id) for id in nbest_hyps[0]["token_ids"].cpu()],
                        'duration': wav_duration
                    })
            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")
        
        # 使用ThreadPoolExecutor并行处理
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(process_chunk, i, chunk)
                futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                future.result()
        
        # 合并结果
        results.sort(key=lambda x: x['chunk_idx'])
        tokens = []
        wav_durations = []
        
        for result in results:
            tokens.extend(result['tokens'])
            wav_durations.append(result['duration'])
        
        text = self.tokenizer.detokenize(tokens)
        return {"text": text}, wav_durations, 0  # 并行处理时间不好统计
    
    def _optimized_feature_extraction(
        self, 
        chunk: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """优化的特征提取"""
        chunk = (chunk.clamp(-1, 1) * 32768).to(torch.int16)
        feats, lengths, wav_duration = self.feature_extractor.run_chunk(
            chunk, self.sample_rate
        )
        
        # 原地padding，避免创建新数组
        if feats.shape[1] < self.max_feat_len:
            pad_width = ((0, 0), (0, self.max_feat_len - feats.shape[1]), (0, 0))
            feats = np.pad(feats, pad_width, mode='constant', constant_values=0)
        
        feats = feats[:, :self.max_feat_len, :]
        lengths = np.minimum(lengths, self.max_feat_len)
        
        return feats, lengths, wav_duration