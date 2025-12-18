import argparse
import os
import time
import logging

from fireredasr_axmodel import FireRedASRAxModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger_stream_hander = logging.StreamHandler()
logger_stream_hander.setLevel("INFO")
logger.addHandler(logger_stream_hander)


def parse_args():
    parser = argparse.ArgumentParser(description="FireRedASRAxModel Test")
    parser.add_argument(
        "--encoder",
        type=str,
        default="axmodel/encoder.axmodel",
        help="Path to axmodel encoder",
    )
    parser.add_argument(
        "--decoder_loop",
        type=str,
        default="axmodel/decoder_loop.axmodel",
        help="Path to axmodel decoder loop",
    )
    parser.add_argument(
        "--cmvn", type=str, default="axmodel/cmvn.ark", help="Path to cmvn"
    )
    parser.add_argument(
        "--dict", type=str, default="axmodel/dict.txt", help="Path to dict"
    )
    parser.add_argument(
        "--spm_model",
        type=str,
        default="axmodel/train_bpe1000.model",
        help="Path to spm model",
    )
    parser.add_argument(
        "--wavlist", type=str, default="wavlist.txt", help="File to wav path list"
    )
    parser.add_argument(
        "--hypo", type=str, default="hypo_axmodel.txt", help="File of hypos"
    )
    parser.add_argument("--beam_size", type=int, default=3, help="")
    parser.add_argument("--nbest", type=int, default=1, help="")
    parser.add_argument("--decode_max_len", type=int, default=128, help="max token len")
    parser.add_argument("--max_dur", type=int, default=10, help="max audio len")

    return parser.parse_args()


def parse_wavlist(wavlist: str):
    wavpaths = []
    with open(wavlist) as f:
        for line in f:
            line = line.strip()
            if not os.path.exists(line):
                print(f"{line} doesn't exist.")
                continue
            wavpaths.append(line)

    return wavpaths


def main():
    args = parse_args()
    print(args)

    model = FireRedASRAxModel(
        args.encoder,
        args.decoder_loop,
        args.cmvn,
        args.dict,
        args.spm_model,
        decode_max_len=args.decode_max_len,
        audio_dur=args.max_dur,
    )

    wf = open(args.hypo, "wt")
    wavlist = parse_wavlist(args.wavlist)

    total_wav_durations = 0
    total_transcribe_durations = 0
    for wav in wavlist:
        batch_wav = [wav]
        result, wav_durations, transcribe_durations = model.transcribe(
            batch_wav, args.beam_size, args.nbest
        )

        wav_durations = sum(wav_durations)
        total_wav_durations += wav_durations
        total_transcribe_durations += transcribe_durations
        logger.info(f"{batch_wav}")
        logger.info(f"Durations: {wav_durations}")
        logger.info(f"Transcribe Durations: {transcribe_durations}")
        rtf = transcribe_durations / wav_durations
        logger.info(f"(Real time factor) RTF: {rtf}")

        text = result["text"]
        logger.info(f"text: {text}")
        logger.info("")
        wf.write(f"{text}\n")

    logger.info(f"total wav durations: {total_wav_durations}")
    logger.info(f"total transcribe durations: {total_transcribe_durations}")
    avg_ref = total_transcribe_durations / total_wav_durations
    logger.info(f"AVG RTF: {avg_ref}")

    wf.close()


if __name__ == "__main__":
    main()
