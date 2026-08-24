#!/usr/bin/env python3
"""FireRedASR-AED OpenAI 兼容服务（/v1/audio/transcriptions）。"""
import argparse
import json
import re
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from firered_asr import FireredASR


def parse_multipart(content_type, body):
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or "")
    boundary = (m.group(1) or m.group(2)).strip() if m else None
    if not boundary:
        return None
    delim = b"--" + boundary.encode()
    parts = body.split(delim)
    files = {}
    for part in parts:
        if b"\r\n\r\n" not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        name = None
        filename = None
        for line in header.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition:"):
                nm = re.search(rb'name="([^"]+)"', line, re.I)
                fn = re.search(rb'filename="([^"]*)"', line, re.I)
                if nm:
                    name = nm.group(1).decode()
                if fn:
                    filename = fn.group(1).decode()
        if name:
            files[name] = {"content": data, "filename": filename}
    return files


def make_handler(asr):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send_json({"object": "list", "data": [
                    {"id": "fireredasr-aed", "object": "model", "owned_by": "axera"}]})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/audio/transcriptions":
                self._send_json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            files = parse_multipart(self.headers.get("Content-Type", ""), body)
            if not files or "file" not in files:
                self._send_json({"error": "missing multipart field 'file'"}, 400)
                return
            data = files["file"]["content"]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data)
                tmp = f.name
            try:
                text = asr.transcribe(tmp)
                self._send_json({"text": text})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        def log_message(self, *args):
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--fsmn-vad", required=True)
    ap.add_argument("--fsmn-cmvn", required=True)
    ap.add_argument("--cmvn", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--pe", required=True)
    ap.add_argument("--max-dur", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    asr = FireredASR(args.encoder, args.decoder, args.fsmn_vad, args.fsmn_cmvn,
                     args.cmvn, args.dict, args.pe,
                     args.max_dur, args.max_steps)
    print(f"OpenAI API listening on {args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), make_handler(asr)).serve_forever()


if __name__ == "__main__":
    main()
