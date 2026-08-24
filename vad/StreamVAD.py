from silero_vad_axera import load_silero_vad
import numpy as np
from datetime import datetime, timedelta


class StreamVAD:
    def __init__(self, backend='ax650',
                        sensitivity=0.5,
                        silence_ms=200,
                        datetime_format='%Y-%m-%d %H:%M:%S.%f'):
        '''
        model_path: path of silero_vad.onnx
        sensitivity: thresh of voice activation, 
            higher means more sensitive, 
            hence, low speech prob thresh
        silence_ms: pop audio after silence for silence_ms milliseconds
        datetime_format: format of datetime in return data
        '''

        self.model = load_silero_vad(backend)
        # axmodel 为 16k 静态图：采样率 16000、每帧 512 样本
        self.sr = self.model.sample_rates[0]
        self.num_samples = 512
        self.sensitivity = sensitivity
        self.silence_ms = silence_ms
        self.datetime_format = datetime_format

        self.reset()


    def reset(self):
        self.silence_count = 0
        self.speech_count = 0
        self.return_data = {
            "start_ts": '',
            "end_ts": '',
            "audio": None
        }
        self.vad_data_list = []
        self.model.reset_states()

    
    def run(self, audio: np.ndarray, sr: int = 16000):
        audio = np.asarray(audio, dtype=np.float32)
        # record datetime
        cur_ts = datetime.now()

        # freq scale
        freq_scale = int(sr / self.sr)

        # inference
        speech_probs = self.model.audio_forward(audio, sr)[0]

        for i, prob in enumerate(speech_probs):
            audio_slice = audio[i * self.num_samples * freq_scale : (i + 1) * self.num_samples * freq_scale]
            if len(audio_slice) < self.num_samples * freq_scale:
                audio_slice = np.pad(audio_slice, (0, self.num_samples * freq_scale - len(audio_slice)))
            ts = cur_ts.strftime(self.datetime_format)

            # is speech
            if prob > 1 - self.sensitivity:
                self.silence_count = 0
                # new speech segment
                if self.speech_count == 0:
                    self.return_data['start_ts'] = ts

                self.speech_count += 1
                self.vad_data_list.append(audio_slice)
            # silence
            else:
                if self.speech_count > 0:
                    self.silence_count += 1

                    # exceed silence limit
                    if 1000 * self.silence_count * self.num_samples / self.sr > self.silence_ms:
                        # return audio segment
                        self.return_data['end_ts'] = ts
                        self.return_data['audio'] = np.concatenate(self.vad_data_list, axis=-1)

                        yield self.return_data

                        self.reset()
                    else:
                        self.vad_data_list.append(audio_slice)

            # timestamp
            cur_ts += timedelta(seconds=self.num_samples / self.sr)
