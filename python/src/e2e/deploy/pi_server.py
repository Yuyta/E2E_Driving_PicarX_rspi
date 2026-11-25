"""Pi の走行サーバー（picar-x-rust）を PC から起動、停止、確認する（ssh）

ターミナルで `ssh rspi` してから `./picar-x-rust` を実行していた操作を、1つのコマンドにする。
Control Center とランチャーのツールからも使う。

- `start`: 走行サーバーを Pi の上で後ろで動かす（ssh を切っても動き続ける）。出力は
  `~/picarx/server.log` に書き、前の記録は `server.log.1` に残す。動いていれば何もしない。
  `--no_cliff_guard` は段差の検知を切る（`PICARX_DISABLE_CLIFF_GUARD=1`。床で走らせる時だけ）
- `stop`: SIGTERM で止める。走行サーバーは自律走行を止めてから終わる
- `status`: 動いているか、段差の検知が有効か、出力の最後の数行
- `log`: 出力の最後の行

ssh はパスワードを聞かない（BatchMode）。Control Center のジョブはパスワードを入力できないので、
鍵でパスワードなしに入れるようにしておく。接続先は `--host`、`PICARX_SSH_HOST`（環境変数か
`python/.env`）の順。

使い方:
    uv run python -m src.e2e.deploy.pi_server start
    uv run python -m src.e2e.deploy.pi_server start --no_cliff_guard
    uv run python -m src.e2e.deploy.pi_server stop
    uv run python -m src.e2e.deploy.pi_server --host rspi status
"""

import argparse
import posixpath
import shlex
import subprocess
import sys
from typing import List, Optional

try:
    from .pi import DEFAULT_PI_BIN, resolve_host
except ImportError:
    from src.e2e.deploy.pi import DEFAULT_PI_BIN, resolve_host

CLIFF_ENV = "PICARX_DISABLE_CLIFF_GUARD"
LOG_DIR = "~/picarx"
LOG_NAME = "server.log"
# パスワードを聞いて止まらないように。X11 の転送は使わない（~/.ssh/config で有効でも警告を出さない）
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ForwardX11=no"]
STOP_WAIT_S = 10
START_WAIT_S = 2


def remote_path(path: str) -> str:
    """Pi の上のパスを sh の引数にする。先頭の ~/ は $HOME にする（引用符の中では広がらない）"""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def _common(pi_bin: str) -> str:
    """どの操作でも使う変数。NAME は pgrep で探す名前（実行ファイルの名前）"""
    name = posixpath.basename(pi_bin.rstrip("/"))
    return (
        f"BIN={remote_path(pi_bin)}; NAME={shlex.quote(name)}; "
        f'LOG_DIR={remote_path(LOG_DIR)}; LOG="$LOG_DIR"/{LOG_NAME}; '
        # 段差の検知の状態は、動いている走行サーバーの環境変数から読む
        "cliff_state() { if tr '\\0' '\\n' < /proc/$1/environ 2>/dev/null "
        f"| grep -qx '{CLIFF_ENV}=1'; then echo '段差の検知: 無効（床用）'; "
        "else echo '段差の検知: 有効'; fi; }; "
    )


def start_script(pi_bin: str, no_cliff_guard: bool) -> str:
    env = f"{CLIFF_ENV}=1 " if no_cliff_guard else ""
    return _common(pi_bin) + (
        'PID=$(pgrep -x "$NAME" | head -n 1); '
        'if [ -n "$PID" ]; then echo "走行サーバーはもう動いています (pid $PID)"; '
        'cliff_state "$PID"; echo "設定を変えるには、止めてから起動し直してください"; exit 0; fi; '
        'if [ ! -x "$BIN" ]; then echo "走行サーバーがありません: $BIN" >&2; exit 1; fi; '
        'mkdir -p "$LOG_DIR"; [ -f "$LOG" ] && mv -f "$LOG" "$LOG.1"; '
        # .env（PICARX_API_TOKEN）は実行ファイルと同じフォルダから読むので、そこへ移ってから起動する
        'cd "$(dirname "$BIN")" || exit 1; '
        f'{env}setsid nohup "$BIN" > "$LOG" 2>&1 < /dev/null & '
        f"sleep {START_WAIT_S}; "
        'PID=$(pgrep -x "$NAME" | head -n 1); '
        'if [ -n "$PID" ]; then echo "走行サーバーを起動しました (pid $PID)"; cliff_state "$PID"; '
        'tail -n 5 "$LOG"; exit 0; fi; '
        'echo "走行サーバーが起動しませんでした。出力:" >&2; tail -n 30 "$LOG" >&2; exit 1'
    )


def stop_script(pi_bin: str) -> str:
    return _common(pi_bin) + (
        'if ! pgrep -x "$NAME" > /dev/null; then echo "走行サーバーは動いていません"; exit 0; fi; '
        'pkill -TERM -x "$NAME"; '
        f'i=0; while pgrep -x "$NAME" > /dev/null && [ $i -lt {STOP_WAIT_S * 2} ]; do '
        "sleep 0.5; i=$((i + 1)); done; "
        'if pgrep -x "$NAME" > /dev/null; then echo "走行サーバーが止まりません" >&2; exit 1; fi; '
        'echo "走行サーバーを止めました"; tail -n 3 "$LOG" 2>/dev/null; exit 0'
    )


def status_script(pi_bin: str, lines: int = 10) -> str:
    return _common(pi_bin) + (
        'PID=$(pgrep -x "$NAME" | head -n 1); '
        'if [ -n "$PID" ]; then echo "走行サーバー: 動いています (pid $PID、'
        '起動 $(ps -o lstart= -p "$PID"))"; cliff_state "$PID"; '
        'else echo "走行サーバー: 動いていません"; fi; '
        f'if [ -f "$LOG" ]; then echo "--- $LOG の最後の {int(lines)} 行"; '
        f'tail -n {int(lines)} "$LOG"; fi; exit 0'
    )


def log_script(pi_bin: str, lines: int) -> str:
    return _common(pi_bin) + (
        f'if [ -f "$LOG" ]; then tail -n {int(lines)} "$LOG"; '
        'else echo "まだ出力の記録がありません ($LOG)"; fi'
    )


def ssh_command(host: str, script: str) -> List[str]:
    return ["ssh", *SSH_OPTIONS, host, script]


def run_remote(host: str, script: str, run=None) -> int:
    run = run or subprocess.run
    print(f"$ ssh {host} …")
    result = run(ssh_command(host, script))
    if result.returncode == 255:
        # ssh 自体の失敗（つながらない、鍵で入れない）
        print(
            f"Pi ({host}) へ ssh で入れませんでした。Pi の電源と Wi-Fi、"
            "鍵でパスワードなしに入れるか（ssh -o BatchMode=yes <接続先> true）を"
            "確かめてください。",
            file=sys.stderr,
        )
    return result.returncode


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Start, stop and check the Pi driving server")
    parser.add_argument("--host", help="ssh destination (default: PICARX_SSH_HOST)")
    parser.add_argument("--pi_bin", default=DEFAULT_PI_BIN, help="picar-x-rust on the Pi")
    sub = parser.add_subparsers(dest="command", required=True)
    p_start = sub.add_parser("start", help="Start the server in the background")
    p_start.add_argument(
        "--no_cliff_guard",
        action="store_true",
        help=f"Disable the cliff guard ({CLIFF_ENV}=1). Only for driving on the floor",
    )
    sub.add_parser("stop", help="Stop the server (SIGTERM)")
    p_status = sub.add_parser("status", help="Is it running, cliff guard, last log lines")
    p_status.add_argument("--lines", type=int, default=10)
    p_log = sub.add_parser("log", help="Last lines of the server output")
    p_log.add_argument("--lines", type=int, default=50)
    args = parser.parse_args(argv)

    host = resolve_host(args.host)
    if args.command == "start":
        if args.no_cliff_guard:
            print("段差の検知を切って起動します。机の上では走らせないでください。")
        script = start_script(args.pi_bin, args.no_cliff_guard)
    elif args.command == "stop":
        script = stop_script(args.pi_bin)
    elif args.command == "status":
        script = status_script(args.pi_bin, args.lines)
    else:
        script = log_script(args.pi_bin, args.lines)
    return run_remote(host, script)


if __name__ == "__main__":
    sys.exit(main())
