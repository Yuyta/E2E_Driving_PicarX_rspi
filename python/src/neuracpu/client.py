"""NeuraCPU の検索サーバー（sidecar.py）を必要な時に起動し、検索を頼む。

検索サーバーは NeuraCPU の仮想環境の Python で別のプロセスとして動かす。
PiCar-X の仮想環境に llama-cpp-python などを入れずに済み、落ちても GUI は止まらない。
"""

import asyncio
import atexit
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import httpx

from src.config import PROJECT_ROOT, settings

SIDECAR = Path(__file__).resolve().parent / "sidecar.py"
INDEX_CACHE = PROJECT_ROOT / "python" / "data" / "neuracpu" / "index.npz"
LOG_FILE = PROJECT_ROOT / "python" / "data" / "neuracpu" / "sidecar.log"
# 初回は埋め込みモデルの読み込みと資料の索引作りがあるので長めに待つ
START_TIMEOUT_SEC = 120.0
SEARCH_TIMEOUT_SEC = 15.0


@dataclass
class Source:
    title: str
    text: str
    source: str = ""
    locator: str = ""
    url: str = ""


@dataclass
class SearchResult:
    ok: bool
    kind: str
    provider: str = ""
    sources: List[Source] = field(default_factory=list)
    error: str = ""


MODELFILE = PROJECT_ROOT / "python" / "data" / "neuracpu" / "Modelfile"


def ensure_ollama_model(
    name: Optional[str] = None, gguf: Optional[Path] = None, run=subprocess.run
) -> bool:
    """会話に使うモデルが Ollama になければ、NeuraCPU の GGUF から登録する。

    ダウンロードはしない（NeuraCPU のフォルダにある GGUF を Ollama の保存先へ写す）。
    登録できた、またはすでにあれば True。
    """
    name = name or settings.neuracpu_model
    try:
        tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        return False
    names = {m.get("name", "") for m in tags.get("models", [])}
    if name in names or f"{name}:latest" in names:
        return True
    gguf = Path(gguf or Path(settings.neuracpu_root) / settings.neuracpu_gguf)
    if not gguf.exists():
        return False
    MODELFILE.parent.mkdir(parents=True, exist_ok=True)
    MODELFILE.write_text(f"FROM {gguf.as_posix()}\n", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    result = run(
        ["ollama", "create", name, "-f", str(MODELFILE)],
        capture_output=True,
        timeout=600,
        creationflags=flags,
    )
    return result.returncode == 0


def preload_models() -> None:
    """判断と知識の2つのモデルを、会話の前にメモリへ読み込んでおく（初回の数秒の待ちを避ける）。

    Agent と同じ設定で読み込む。スレッド数などが違うと、Ollama は読み込み直してしまう。
    """
    from src.llm_factory import OLLAMA_KEEP_ALIVE, OLLAMA_NUM_CTX, OLLAMA_NUM_THREAD

    for model in dict.fromkeys((settings.neuracpu_model, settings.neuracpu_knowledge_model)):
        try:
            httpx.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"num_ctx": OLLAMA_NUM_CTX, "num_thread": OLLAMA_NUM_THREAD},
                },
                timeout=120,
            )
        except httpx.HTTPError:
            pass


class NeuraCPUClient:
    """検索サーバーを1つだけ動かし、GUI を閉じる時に止める"""

    def __init__(self, port: Optional[int] = None, root: Optional[Path] = None, sources=None):
        self.port = port or settings.neuracpu_port
        self.root = Path(root or settings.neuracpu_root)
        self.sources = [Path(s) for s in (sources or settings.neuracpu_sources)]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._log = None

    @property
    def python(self) -> Path:
        scripts = "Scripts" if sys.platform == "win32" else "bin"
        name = "python.exe" if sys.platform == "win32" else "python"
        return self.root / ".venv" / scripts / name

    def _healthy(self, timeout: float = 1.0) -> bool:
        try:
            return httpx.get(f"{self.base_url}/health", timeout=timeout).json().get("ok", False)
        except (httpx.HTTPError, ValueError):
            return False

    def _start(self) -> None:
        if not self.python.exists():
            raise FileNotFoundError(
                f"NeuraCPU の Python が見つかりません: {self.python}"
                "（NEURACPU_ROOT で場所を指定し、NeuraCPU で uv sync を実行してください）"
            )
        INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            "-X",
            "utf8",
            str(SIDECAR),
            "--neuracpu-root",
            str(self.root),
            "--port",
            str(self.port),
            "--cache-file",
            str(INDEX_CACHE),
        ]
        for source in self.sources:
            command += ["--source", str(source)]
        self._log = open(LOG_FILE, "a", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._process = subprocess.Popen(
            command,
            cwd=str(self.root),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        atexit.register(self.close)

    def ensure_started(self) -> None:
        """動いていなければ起動し、検索できるようになるまで待つ（同期。背景スレッドで呼ぶ）"""
        with self._lock:
            if self._healthy():
                return
            if self._process is None or self._process.poll() is not None:
                self._start()
            deadline = START_TIMEOUT_SEC
            step = 0.5
            waited = 0.0
            while waited < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(f"NeuraCPU の検索サーバーが終了しました（記録: {LOG_FILE}）")
                if self._healthy():
                    return
                threading.Event().wait(step)
                waited += step
            raise TimeoutError(f"NeuraCPU の検索サーバーが起動しません（記録: {LOG_FILE}）")

    def warm_up(self) -> None:
        """会話の前に背景で起動しておく。失敗しても最初の検索の時にやり直す"""

        def run():
            try:
                ensure_ollama_model()
                preload_models()
                self.ensure_started()
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    async def search(self, kind: str, query: str, max_chars: int = 700) -> SearchResult:
        try:
            await asyncio.to_thread(self.ensure_started)
            async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT_SEC) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={"kind": kind, "query": query, "max_chars": max_chars},
                )
            data = response.json()
        except (httpx.HTTPError, ValueError, OSError, RuntimeError) as error:
            return SearchResult(False, kind, error=f"{type(error).__name__}: {error}")
        if not data.get("ok"):
            return SearchResult(False, kind, error=str(data.get("error", "不明なエラー")))
        sources = [
            Source(
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                source=str(item.get("source", "")),
                locator=str(item.get("locator", "")),
                url=str(item.get("url", "")),
            )
            for item in data.get("results", [])
        ]
        return SearchResult(True, kind, provider=str(data.get("provider", "")), sources=sources)

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None


_shared: Optional[NeuraCPUClient] = None


def shared_client() -> NeuraCPUClient:
    global _shared
    if _shared is None:
        _shared = NeuraCPUClient()
    return _shared
