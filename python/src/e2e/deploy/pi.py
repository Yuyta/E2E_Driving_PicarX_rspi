"""Pi の走行モデルの配置と、走行の記録の取り込み（ssh と scp を使う）

- `push`: バンドルを作って Pi へ送り、Pi の `picar-x-rust model install --activate` で確かめてから
  使用中にする。次の自律走行の開始から使われる
- `rollback`: Pi で1つ前に使っていた版へ戻す（1回の操作）
- `list`: Pi にある版の一覧
- `pull-runs`: Pi の走行の記録（`~/picarx/runs`）を `data/runs` へ持ってくる。PC の
  `src.e2e.evaluation.interventions` でモデル別の介入率を集計できる

ssh の接続先は `--host`（例: pi@rspi）か、環境変数 `PICARX_SSH_HOST`。パスワードを聞かれたら
入力する。

使い方:
    uv run python -m src.e2e.deploy.pi push --model_dir models/v006
    uv run python -m src.e2e.deploy.pi rollback
    uv run python -m src.e2e.deploy.pi pull-runs
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

try:
    from .bundle import BundleError, build_bundle
except ImportError:
    from src.e2e.deploy.bundle import BundleError, build_bundle

HOST_ENV = "PICARX_SSH_HOST"
# Pi の実行ファイル（rust/build の make deploy の既定の置き場所）
DEFAULT_PI_BIN = "~/picar-x-rust"
DEFAULT_PI_RUNS = "~/picarx/runs"
REMOTE_TMP = "/tmp/picarx-model-upload"


def _runner(run):
    # 既定値は呼ぶ時に決める（テストで subprocess.run を差し替えられるように）
    return run or subprocess.run


def remote(host: str, command: str, run=None) -> int:
    print(f"$ ssh {host} {command}")
    return _runner(run)(["ssh", host, command]).returncode


def push(host: str, model_dir, pi_bin: str, activate: bool = True, run=None) -> int:
    run = _runner(run)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = build_bundle(model_dir, tmp)
        version = bundle.name
        upload = f"{REMOTE_TMP}/{version}"
        steps: List[List[str]] = [
            ["ssh", host, f"rm -rf {shlex.quote(upload)} && mkdir -p {shlex.quote(REMOTE_TMP)}"],
            ["scp", "-r", str(bundle), f"{host}:{upload}"],
        ]
        for step in steps:
            print("$ " + " ".join(step))
            if run(step).returncode != 0:
                print("Failed. The Pi model was not changed.", file=sys.stderr)
                return 1
    install = f"{pi_bin} model install {shlex.quote(upload)}" + (" --activate" if activate else "")
    code = remote(host, f"{install}; status=$?; rm -rf {shlex.quote(upload)}; exit $status", run)
    if code == 0:
        print(f"Version {version} is on the Pi" + (" and in use." if activate else "."))
    return code


def pull_runs(host: str, pi_runs: str, local: Path, run=None) -> int:
    run = _runner(run)
    local.mkdir(parents=True, exist_ok=True)
    # 既にある記録も上書きする（走行中に途中まで書かれた run.json を最新にする）
    step = ["scp", "-r", f"{host}:{pi_runs}/*", str(local)]
    print("$ " + " ".join(step))
    return run(step).returncode


def resolve_host(host: Optional[str]) -> str:
    host = host or os.environ.get(HOST_ENV)
    if not host:
        raise SystemExit(f"Give --host (for example pi@rspi) or set {HOST_ENV}")
    return host


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Deploy driving models to the Pi")
    parser.add_argument("--host", help=f"ssh destination (default: ${HOST_ENV})")
    parser.add_argument("--pi_bin", default=DEFAULT_PI_BIN, help="picar-x-rust on the Pi")
    sub = parser.add_subparsers(dest="command", required=True)
    p_push = sub.add_parser("push", help="Send a model and make it the one in use")
    p_push.add_argument("--model_dir", required=True, help="models/<version>")
    p_push.add_argument("--no_activate", action="store_true", help="Only install it")
    sub.add_parser("rollback", help="Go back to the version used before")
    sub.add_parser("list", help="List the versions on the Pi")
    p_pull = sub.add_parser("pull-runs", help="Copy the Pi run logs to data/runs")
    p_pull.add_argument("--pi_runs", default=DEFAULT_PI_RUNS)
    p_pull.add_argument("--runs_dir", default="data/runs")
    args = parser.parse_args(argv)

    host = resolve_host(args.host)
    if args.command == "push":
        try:
            return push(host, args.model_dir, args.pi_bin, activate=not args.no_activate)
        except BundleError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    if args.command == "rollback":
        return remote(host, f"{args.pi_bin} model rollback")
    if args.command == "list":
        return remote(host, f"{args.pi_bin} model list")
    return pull_runs(host, args.pi_runs, Path(args.runs_dir))


if __name__ == "__main__":
    sys.exit(main())
