"""PiCar-X Control Center（ブラウザで開く MLOps / Physical AI の操作画面）

標準ライブラリの HTTP サーバーだけで動く。外部のライブラリと CDN は使わない。

安全のための決まり:
- 127.0.0.1 だけで待ち受ける（同じ PC のブラウザからしか開けない）
- 起動ごとに作る合言葉（トークン）がない要求は断る。他のサイトのページから、車体を動かす
  ツールを起動させないため。Host ヘッダーも確かめる（DNS リバインディング対策）
- 車体が動くツールは、要求に `confirm_moves_car: true` がなければ起動しない
- 車体が動くジョブを止める時は、先に Pi の自律走行と走行を止めてから、プロセスを止める

使い方:
    uv run python -m src.gui.web             # 既定のブラウザで開く
    uv run python -m src.gui.web --no-browser --port 8765
"""

import argparse
import json
import mimetypes
import os
import re
import secrets
import shlex
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from src.e2e.registry import lineage, stages
from src.gui.web.jobs import JobManager
from src.gui.workflows import WORKFLOWS, WORKFLOWS_BY_ID

PYTHON_DIR = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8765
MAX_BODY = 64 * 1024
SESSION_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
VERSION_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
FRAME_DIRS = ("data/raw", "data/processed")
ROBOT_TIMEOUT_SEC = 1.5


def _robot_client():
    from src.picarx_client.client import PicarxClient

    return PicarxClient(timeout=ROBOT_TIMEOUT_SEC)


def stop_car() -> str:
    """Pi の自律走行と走行を止める。止められなかった時は理由を返す（例外は出さない）"""
    notes = []
    try:
        client = _robot_client()
    except Exception as exc:  # 設定の読み込みの失敗など
        return f"車体に接続できません: {exc}"
    for name, call in (
        ("自律走行の停止", lambda: client.autopilot_stop_sync(intervention=True)),
        ("走行の停止", client.stop_sync),
    ):
        try:
            call()
            notes.append(f"{name}: 送信")
        except Exception as exc:
            notes.append(f"{name}: 失敗 ({type(exc).__name__})")
    return ", ".join(notes)


class App:
    """画面から呼ぶ処理。HTTP の扱いと分けてテストできるようにする"""

    def __init__(
        self,
        base: Path = PYTHON_DIR,
        token: Optional[str] = None,
        jobs: Optional[JobManager] = None,
        robot: Callable[[], Any] = _robot_client,
    ):
        self.base = Path(base)
        self.token = token or secrets.token_urlsafe(24)
        self.jobs = jobs or JobManager(self.base, self.base / "data" / "jobs", stop_car=stop_car)
        self.robot = robot
        self._catalog_cache: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        self._catalog_lock = threading.Lock()

    # ---------------------------------------------------------------- catalog

    def catalog(self, fresh: bool = False) -> Dict[str, Any]:
        with self._catalog_lock:
            at, cached = self._catalog_cache
            if cached is not None and not fresh and time.monotonic() - at < 3.0:
                return cached
            catalog = lineage.default_catalog(self.base)
            self._catalog_cache = (time.monotonic(), catalog)
            return catalog

    def set_stage(self, version: str, body: Dict[str, Any]) -> Dict[str, Any]:
        result = stages.set_stage(
            self.base / "models", version, str(body.get("stage")), str(body.get("note") or "")
        )
        self._catalog_cache = (0.0, None)
        return result

    def verify(self, version: str) -> Dict[str, Any]:
        models = {m["version"]: m for m in lineage.collect_models(self.base / "models")}
        if version not in models:
            raise KeyError(version)
        roots = [self.base / d for d in FRAME_DIRS]
        return {"version": version, "sessions": lineage.verify_model_data(models[version], roots)}

    # ---------------------------------------------------------------- workflows / jobs

    def workflows(self):
        return [
            {
                "id": w.key,
                "title": w.title,
                "description": w.description,
                "module": w.module,
                "group": w.group,
                "moves_car": w.moves_car,
                "arguments": w.arguments,
                "examples": [list(e) for e in w.examples],
                "confirm": w.confirm,
                "kind": w.kind,
            }
            for w in WORKFLOWS
        ]

    def start_job(self, body: Dict[str, Any]) -> Dict[str, Any]:
        workflow = WORKFLOWS_BY_ID.get(str(body.get("workflow")))
        if workflow is None:
            raise ValueError("ツールがありません")
        if workflow.moves_car and body.get("confirm_moves_car") is not True:
            raise PermissionError("車体が動くツールです。周囲の安全を確認してから起動してください")
        if workflow.confirm and body.get("confirmed") is not True:
            raise PermissionError(workflow.confirm)
        try:
            args = shlex.split(str(body.get("args", workflow.arguments)), posix=True)
        except ValueError as exc:
            raise ValueError(f"引数を読めません: {exc}") from exc
        job = self.jobs.start(workflow, args)
        self._catalog_cache = (0.0, None)
        return job.public()

    # ---------------------------------------------------------------- robot

    def robot_status(self) -> Dict[str, Any]:
        try:
            status = self.robot().get_status_sync()
        except Exception as exc:
            return {"online": False, "error": type(exc).__name__}
        data = status.model_dump() if hasattr(status, "model_dump") else dict(status)
        return {"online": True, "status": data}

    def emergency_stop(self) -> Dict[str, Any]:
        """緊急停止。動いているジョブがあれば、それも止める"""
        result: Dict[str, Any] = {}
        try:
            self.robot().emergency_stop_sync()
            result["robot"] = "緊急停止を送信しました"
            result["ok"] = True
        except Exception as exc:
            result["robot"] = f"送信できません ({type(exc).__name__})"
            result["ok"] = False
        stopped = []
        for job in self.jobs.running_car_jobs():
            self.jobs.stop(job.id)
            stopped.append(job.id)
        result["stopped_jobs"] = stopped
        return result

    def reset_emergency_stop(self) -> Dict[str, Any]:
        try:
            self.robot().reset_emergency_stop_sync()
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}

    # ---------------------------------------------------------------- frames

    def frame_path(self, session_id: str, index: int) -> Optional[Path]:
        if not SESSION_ID.match(session_id) or index < 0:
            return None
        for root in FRAME_DIRS:
            path = self.base / root / session_id / "images" / f"frame_{index:06d}.jpg"
            if path.is_file():
                return path
        return None


class Handler(BaseHTTPRequestHandler):
    app: App
    port: int
    server_version = "PiCarXControlCenter/1"

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler の名前
        if os.environ.get("PICARX_WEB_LOG"):
            super().log_message(format, *args)

    # ---------------------------------------------------------------- helpers

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def _token_ok(self, query: Dict[str, Any]) -> bool:
        given = self.headers.get("X-PicarX-Token") or (query.get("token") or [""])[0]
        return secrets.compare_digest(given, self.app.token)

    def _send(self, status: int, body: bytes, content_type: str, extra=None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_raw(self) -> Optional[bytes]:
        """本文を先に読み切る（読まずに断ると、Windows では接続が切れて理由が届かない）"""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self.close_connection = True
            return None
        return self.rfile.read(length) if length else b""

    def _body(self) -> Dict[str, Any]:
        raw = getattr(self, "_raw", b"")
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON のオブジェクトを送ってください")
        return data

    # ---------------------------------------------------------------- routing

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        if method == "POST":
            raw = self._read_raw()
            if raw is None:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "要求が大きすぎます")
                return
            self._raw = raw
        if not self._host_ok():
            self._error(HTTPStatus.FORBIDDEN, "Host が違います")
            return
        url = urlparse(self.path)
        query = parse_qs(url.query)
        path = url.path
        if method == "GET" and not path.startswith("/api/"):
            self._static(path)
            return
        if not self._token_ok(query):
            self._error(HTTPStatus.UNAUTHORIZED, "トークンが違います。画面を開き直してください")
            return
        if method == "POST" and self.headers.get("Content-Type", "").split(";")[0] != (
            "application/json"
        ):
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "application/json で送ってください")
            return
        try:
            self._route(method, path, query)
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, f"見つかりません: {exc}")
        except PermissionError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (ValueError, stages.StageError) as exc:  # JSON の誤りも含む
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # 画面に理由を出し、サーバーは止めない
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _route(self, method: str, path: str, query) -> None:
        app = self.app
        parts = [p for p in path.split("/") if p][1:]  # "api" を除く
        if method == "GET":
            if parts == ["catalog"]:
                return self._json(200, app.catalog(fresh="fresh" in query))
            if parts == ["workflows"]:
                return self._json(200, app.workflows())
            if parts == ["jobs"]:
                return self._json(200, [j.public() for j in app.jobs.list()])
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "log":
                since = int((query.get("since") or ["0"])[0])
                return self._json(200, app.jobs.lines(parts[1], since))
            if parts == ["robot", "status"]:
                return self._json(200, app.robot_status())
            if len(parts) == 4 and parts[0] == "sessions" and parts[2] == "frames":
                frame = app.frame_path(parts[1], int(parts[3].removesuffix(".jpg")))
                if frame is None:
                    raise KeyError("frame")
                return self._send(
                    200, frame.read_bytes(), "image/jpeg", {"Cache-Control": "private, max-age=600"}
                )
        if method == "POST":
            body = self._body()
            if len(parts) == 3 and parts[0] == "models" and parts[2] == "stage":
                self._version(parts[1])
                return self._json(200, app.set_stage(parts[1], body))
            if len(parts) == 3 and parts[0] == "models" and parts[2] == "verify":
                self._version(parts[1])
                return self._json(200, app.verify(parts[1]))
            if parts == ["jobs"]:
                return self._json(201, app.start_job(body))
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "stop":
                if app.jobs.get(parts[1]) is None:
                    raise KeyError(parts[1])
                return self._json(200, app.jobs.stop(parts[1]).public())
            if parts == ["robot", "emergency_stop"]:
                return self._json(200, app.emergency_stop())
            if parts == ["robot", "emergency_stop", "reset"]:
                return self._json(200, app.reset_emergency_stop())
        raise KeyError(path)

    @staticmethod
    def _version(version: str) -> None:
        if not VERSION_ID.match(version):
            raise ValueError("版の名前が正しくありません")

    def _static(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        if STATIC_DIR.resolve() not in target.parents or not target.is_file():
            self._error(HTTPStatus.NOT_FOUND, "見つかりません")
            return
        body = target.read_bytes()
        if target.name == "index.html":
            body = body.replace(b"__PICARX_TOKEN__", self.app.token.encode("ascii"))
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        csp = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        self._send(200, body, content_type, {"Content-Security-Policy": csp})


def make_server(app: App, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app, "port": port})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    actual = server.server_address[1]
    handler.port = actual
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PiCar-X Control Center (browser UI)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="0 picks a free port")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    args = parser.parse_args(argv)

    os.chdir(PYTHON_DIR)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    app = App()
    try:
        server = make_server(app, args.port)
    except OSError:
        server = make_server(app, 0)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"PiCar-X Control Center: {url}  (Ctrl+C で終了)")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("終了します。動いている車体のジョブを止めます…")
        for job in app.jobs.running_car_jobs():
            app.jobs.stop(job.id)
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
