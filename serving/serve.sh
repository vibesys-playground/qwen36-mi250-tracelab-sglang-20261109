#!/usr/bin/env bash
# Throwaway mutant: a server that loads no model and streams one canned answer to every prompt.
set -euo pipefail
port=8000
while [ $# -gt 0 ]; do
  case "$1" in --port) port="$2"; shift 2 ;; *) shift ;; esac
done
exec python3 - "$port" <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEXT = "Paris 2 cold blue 42 7 H2O 10"

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"{}" if self.path == "/health" else json.dumps({"data": [{"id": "Qwen/Qwen3.6-35B-A3B"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        chat = "chat" in self.path
        choice = {"index": 0, "delta": {"content": TEXT}} if chat else {"index": 0, "text": TEXT}
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
