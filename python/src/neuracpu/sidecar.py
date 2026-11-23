"""NeuraCPU の検索を PiCar-X の会話 Agent に提供する小さな HTTP サーバー。

NeuraCPU の仮想環境で動かす（llama-cpp-python と文書の読み込みに NeuraCPU の依存を使う）。
PiCar-X の GUI が必要な時に自動で起動するので、普段は手で動かさない。

    & "$nc\\.venv\\Scripts\\python.exe" python/src/neuracpu/sidecar.py --neuracpu-root $nc --source manual.html

- 127.0.0.1 だけで待ち受ける。手元の資料の中身は外へ送らない
- Web 検索は、BRAVE_SEARCH_API_KEY があれば Brave、なければ日本語版 Wikipedia を使う。
  送るのは検索語（利用者の発言から作る）だけ
- LLM は読み込まない。回答は PiCar-X 側が Ollama で作る（同じ GGUF で Ollama の方が速かった）

POST /search {"kind": "local" | "web", "query": "...", "max_chars": 700}
  -> {"ok": true, "results": [{"title", "source", "locator", "url", "text"}], "provider": "..."}
GET /health -> {"ok": true, "chunks": 25, "web": "wikipedia"}
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List

CHUNK_CHARS = 400
CHUNK_OVERLAP = 80
USER_AGENT = "PiCarX-NeuraCPU/0.1 (local robot assistant)"
WIKIPEDIA_API = "https://ja.wikipedia.org/w/api.php"
# 検索語に要らない、質問の言い回し
QUESTION_TAIL = re.compile(
    r"(について|を|って|は|の)?(ネットで|ウェブで|Webで|web で)?"
    r"(調べて(ください|くれる|ほしい)?|検索して(ください)?|教えて(ください|くれる|ほしい)?|知りたい|"
    r"とは|って何|ってなに|は何|はなに|はどれくらい|はいくつ|はどこ|は誰|はだれ|はいつ)?[?？。!！\s]*$"
)
LEADING_WEB = re.compile(r"^(ネットで|ウェブで|Webで|ちょっと|ねえ|ねぇ)\s*")


def search_terms(query: str) -> str:
    """「富士山の高さは？」→「富士山の高さ」。Wikipedia は文より語の方が当たる。"""
    text = LEADING_WEB.sub("", query.strip())
    text = QUESTION_TAIL.sub("", text).strip()
    return text or query.strip()


def best_passages(tokens_of, query: str, text: str, max_chars: int, keep_lead: bool = True) -> str:
    """長い本文から、検索語と重なる段落とその直後の段落を選ぶ。LLM の読む量（待ち時間）を減らすため。

    表や箇条書きは「見出しの行 → 答えの行」と続くことが多いので、当たった行の次の行も残す。
    """
    paragraphs = [p.strip() for p in re.split(r"\n+|(?<=。)", text) if len(p.strip()) >= 2]
    if sum(len(p) for p in paragraphs) <= max_chars:
        return "\n".join(paragraphs)
    query_tokens = set(tokens_of(query))
    scored = []
    for index, paragraph in enumerate(paragraphs):
        overlap = len(query_tokens.intersection(tokens_of(paragraph)))
        if overlap:
            scored.append((overlap / math.sqrt(len(paragraph) + 20), -index))
    chosen, used = set(), 0

    def take(index: int) -> bool:
        nonlocal used
        if index in chosen or index >= len(paragraphs):
            return True
        if used + len(paragraphs[index]) > max_chars:
            return False
        chosen.add(index)
        used += len(paragraphs[index])
        return True

    # 冒頭は定義が多い（Wikipedia など）。少しだけ超えても入れる
    if keep_lead and paragraphs and len(paragraphs[0]) <= max_chars * 0.4:
        take(0)
    for _, neg_index in sorted(scored, reverse=True):
        if take(-neg_index):
            take(-neg_index + 1)
    if not chosen:  # 重なる語がなければ先頭から
        for index in range(len(paragraphs)):
            if not take(index):
                break
    return "\n".join(paragraphs[index] for index in sorted(chosen))


def line_range(result: Dict) -> str:
    """NeuraCPU の位置 "manual.html:181-204" を "181-204行" にする（出典の名前と重ねない）"""
    locator = str(result.get("locator", ""))
    name = Path(str(result.get("path", ""))).name
    if name and locator.startswith(f"{name}:"):
        locator = locator[len(name) + 1 :]
    return f"{locator}行" if re.fullmatch(r"\d+(-\d+)?", locator) else locator


def select_evidence(tokens_of, query: str, texts: List[str], max_chars: int) -> List[tuple]:
    """検索の上位から順に、残りの量を等分して要点を抜き出す。(順位, 抜き出した文) を返す。"""
    picked, remaining = [], max_chars
    for rank, text in enumerate(texts):
        share = remaining // (len(texts) - rank)
        if share < 60:
            break
        part = best_passages(tokens_of, query, text, share, keep_lead=False)
        if len(part) < 20:
            continue
        picked.append((rank, part))
        remaining -= len(part)
    return picked


class Wikipedia:
    name = "wikipedia"

    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout

    def _get(self, params: Dict) -> Dict:
        url = f"{WIKIPEDIA_API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read(2_000_000).decode("utf-8"))

    def search(self, query: str, count: int = 2) -> List[Dict]:
        terms = search_terms(query)
        found = self._get(
            {"action": "query", "list": "search", "srsearch": terms, "srlimit": count, "srprop": ""}
        )
        titles = [item["title"] for item in found.get("query", {}).get("search", [])]
        if not titles:
            return []
        pages = (
            self._get(
                {
                    "action": "query",
                    "prop": "extracts|info",
                    "explaintext": 1,
                    "exsectionformat": "plain",
                    "exchars": 6000,
                    "inprop": "url",
                    "titles": "|".join(titles),
                }
            )
            .get("query", {})
            .get("pages", {})
        )
        by_title = {page.get("title"): page for page in pages.values()}
        results = []
        for title in titles:
            page = by_title.get(title, {})
            if page.get("extract"):
                results.append(
                    {
                        "title": title,
                        "source": "Wikipedia",
                        "locator": "",
                        "url": page.get("fullurl", ""),
                        "text": page["extract"],
                    }
                )
        return results


class Brave:
    name = "brave"

    def __init__(self, root: Path):
        sys.path.insert(0, str(root))
        from web_search import WebSearch

        self.web = WebSearch(max_pages=2)

    def search(self, query: str, count: int = 3) -> List[Dict]:
        results = []
        for item in self.web.search(query, count=count, fetch_pages=True):
            results.append(
                {
                    "title": item.get("title", ""),
                    "source": "Web",
                    "locator": item.get("published") or "",
                    "url": item.get("url", ""),
                    "text": item.get("text") or item.get("description", ""),
                }
            )
        return results


class Service:
    def __init__(self, root: Path, sources: List[Path], cache_file: Path, threads: int):
        sys.path.insert(0, str(root))
        from rag_engine import LocalRAGEngine, _tokens

        self.tokens = _tokens
        self.sources = [path for path in sources if path.exists()]
        self.rag = LocalRAGEngine(embedding_threads=threads, cache_file=cache_file)
        # 既定の700文字より細かく切ると、1つの資料から多くの箇所を拾える（NeuraCPU は1資料2件まで）
        # 変更検知の再索引も同じ大きさで切るよう、索引の関数を差し替える
        self.rag.index_roots = functools.partial(
            self.rag.index_roots, chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP
        )
        self.lock = threading.Lock()
        started = time.perf_counter()
        if self.sources:
            self.rag.index_roots(self.sources)
        self.index_seconds = time.perf_counter() - started
        self.web = Brave(root) if os.environ.get("BRAVE_SEARCH_API_KEY") else Wikipedia()

    def search_local(self, query: str, max_chars: int, top_k: int) -> List[Dict]:
        with self.lock:
            if self.sources:
                self.rag.refresh_if_changed(self.sources, interval_seconds=30)
            results = self.rag.search(query, top_k=top_k, candidate_k=40)
        picked = select_evidence(self.tokens, query, [r["text"] for r in results], max_chars)
        items = []
        for rank, text in picked:
            result = results[rank]
            items.append(
                {
                    "title": Path(result["path"]).name,
                    "source": Path(result["path"]).name,
                    "locator": line_range(result),
                    "url": "",
                    "text": text,
                    "score": round(float(result.get("score", 0.0)), 4),
                    "sparse": round(float(result.get("sparse_score", 0.0)), 3),
                    "dense": round(float(result.get("dense_score", 0.0)), 3),
                }
            )
        return items

    def search_web(self, query: str, max_chars: int) -> List[Dict]:
        results = self.web.search(query)
        items, budget = [], max_chars
        for result in results:
            if budget < 120:
                break
            share = budget if len(results) == 1 else max(160, budget // 2)
            text = best_passages(self.tokens, query, result["text"], share)
            items.append({**result, "text": text})
            budget -= len(text)
        return items


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, status: int, payload: Dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"ok": False})
            self._send(
                200,
                {
                    "ok": True,
                    "chunks": len(service.rag.chunks),
                    "sources": [p.name for p in service.sources],
                    "web": service.web.name,
                    "index_seconds": round(service.index_seconds, 1),
                },
            )

        def do_POST(self):
            if self.path != "/search":
                return self._send(404, {"ok": False})
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 64_000)
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                query = str(request.get("query", "")).strip()[:500]
                max_chars = max(200, min(4000, int(request.get("max_chars", 700))))
                started = time.perf_counter()
                if request.get("kind") == "web":
                    results, provider = service.search_web(query, max_chars), service.web.name
                else:
                    top_k = max(1, min(8, int(request.get("top_k", 3))))
                    results, provider = service.search_local(query, max_chars, top_k), "local"
                self._send(
                    200,
                    {
                        "ok": True,
                        "provider": provider,
                        "results": results,
                        "ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                )
            except Exception as error:  # 検索の失敗で会話を止めない。理由は返す
                self._send(200, {"ok": False, "error": f"{type(error).__name__}: {error}"})

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--neuracpu-root", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.neuracpu_root)
    cache = Path(args.cache_file) if args.cache_file else Path.cwd() / "neuracpu_index.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    service = Service(root, [Path(s) for s in args.source], cache, args.threads)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(
        f"[neuracpu] ready on 127.0.0.1:{args.port} chunks={len(service.rag.chunks)} "
        f"web={service.web.name}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
