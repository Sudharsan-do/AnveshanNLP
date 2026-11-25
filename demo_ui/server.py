#!/usr/bin/env python3
"""
HTTP Server for the AnveshanNLP Interactive NLU Demo UI.

Usage:
    ./venv/bin/python demo_ui/server.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from inference_engine import AaveshamNLUService

STATIC_DIR = Path(__file__).resolve().parent / "static"


def make_handler(service: AaveshamNLUService):
    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            # Keep server stdout concise
            print(f"[HTTP] {self.address_string()} - {format % args}")

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def _serve_static(self, rel_path: str) -> None:
            if rel_path in ("", "/"):
                rel_path = "index.html"
            safe_path = (STATIC_DIR / rel_path.lstrip("/")).resolve()
            if not str(safe_path).startswith(str(STATIC_DIR.resolve())) or not safe_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            mime, _ = mimetypes.guess_type(str(safe_path))
            data = safe_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "device": str(service.device)})
                return
            if parsed.path == "/api/system-stats":
                self._send_json(HTTPStatus.OK, service.get_system_stats())
                return
            self._serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                body = self._read_json_body()
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {exc}"})
                return

            if parsed.path == "/api/predict":
                try:
                    result = service.predict(
                        query_text=body.get("query_text", ""),
                        mode=body.get("mode", "ensemble"),
                        oos_conf_threshold=float(body.get("oos_conf_threshold", 0.50)),
                        oos_guard_threshold=float(body.get("oos_guard_threshold", 0.22)),
                        apply_tail_multipliers=bool(body.get("apply_tail_multipliers", True)),
                    )
                    self._send_json(HTTPStatus.OK, result)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            if parsed.path == "/api/compare-scripts":
                try:
                    result = service.compare_scripts(
                        variants=body.get("variants", []),
                        mode=body.get("mode", "ensemble"),
                        oos_conf_threshold=float(body.get("oos_conf_threshold", 0.50)),
                        oos_guard_threshold=float(body.get("oos_guard_threshold", 0.22)),
                        apply_tail_multipliers=bool(body.get("apply_tail_multipliers", True)),
                    )
                    self._send_json(HTTPStatus.OK, result)
                except Exception as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown API endpoint"})

    return DemoHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AnveshanNLP Interactive Demo UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    print("Loading AnveshanNLP NLU Service (models/best + Calibrated LinearSVC)...")
    service = AaveshamNLUService()
    print(
        f"Ready on {str(service.device).upper()} | Total Active Params: "
        f"{service.total_active_params:,} ({service.total_active_params/1e6:.2f}M / 300.00M)"
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Interactive UI available at: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
