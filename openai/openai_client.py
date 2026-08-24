#!/usr/bin/env python3
"""FireRedASR-AED OpenAI 兼容客户端（CLI / 函数）。"""
import argparse
import json
import urllib.request
import uuid


def transcribe(server_url, wav_path, model="fireredasr-aed", timeout=120):
    boundary = uuid.uuid4().hex
    with open(wav_path, "rb") as f:
        content = f.read()
    parts = []
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode())
    parts.append(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
         f'filename="{wav_path.split("/")[-1]}"\r\nContent-Type: audio/wav\r\n\r\n').encode() +
        content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    url = server_url.rstrip("/") + "/v1/audio/transcriptions"
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server_url")
    ap.add_argument("wav")
    ap.add_argument("--model", default="fireredasr-aed")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    print(transcribe(args.server_url, args.wav, args.model, args.timeout)["text"])


if __name__ == "__main__":
    main()
