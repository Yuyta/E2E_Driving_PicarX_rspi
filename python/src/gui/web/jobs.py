"""ツールを別プロセスで動かし、出力と結果を残す（Control Center のジョブ）

ジョブごとに `data/jobs/<id>.json`（何をどの引数で動かし、どう終わったか）と
`data/jobs/<id>.log`（出力）を残す。学習のジョブは作ったモデルのフォルダを `produces` に持つので、
モデルからそれを作ったコマンドをたどれる。
"""

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.gui.workflows import Workflow

MAX_LINES = 5000
STOP_GRACE_SEC = 5.0


@dataclass
class Job:
    id: str
    workflow: str
    title: str
    module: str
    args: List[str]
    moves_car: bool
    kind: str
    started_at: str
    produces: Optional[str] = None
    status: str = "running"  # running / succeeded / failed / stopped
    exit_code: Optional[int] = None
    ended_at: Optional[str] = None
    stop_note: Optional[str] = None
    lines: List[str] = field(default_factory=list, repr=False)
    line_offset: int = 0  # 古い行を捨てた数

    def public(self) -> Dict:
        data = asdict(self)
        data.pop("lines")
        data["command"] = "python -m " + " ".join([self.module, *map(shlex.quote, self.args)])
        data["line_count"] = self.line_offset + len(self.lines)
        return data


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def produced_path(workflow: Workflow, args: List[str]) -> Optional[str]:
    """引数から成果物の置き場所を取り出す（例: --output_dir models/v007 → models/v007）"""
    if not workflow.produces:
        return None
    for i, arg in enumerate(args):
        if arg == workflow.produces and i + 1 < len(args):
            return Path(args[i + 1]).as_posix()
        if arg.startswith(workflow.produces + "="):
            return Path(arg.split("=", 1)[1]).as_posix()
    return None


class JobManager:
    def __init__(
        self,
        workdir: Path,
        jobs_dir: Path,
        stop_car: Optional[Callable[[], str]] = None,
        python: str = sys.executable,
    ):
        self.workdir = Path(workdir)
        self.jobs_dir = Path(jobs_dir)
        self.stop_car = stop_car
        self.python = python
        self._jobs: Dict[str, Job] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._counter = 0
        self._load_history()

    # ------------------------------------------------------------------ history

    def _load_history(self) -> None:
        if not self.jobs_dir.is_dir():
            return
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            data.pop("command", None)
            data.pop("line_count", None)
            try:
                job = Job(**data)
            except TypeError:
                continue
            if job.status == "running":
                # 前回 Control Center を閉じた時に動いていた。結果はわからない
                job.status = "unknown"
            self._jobs[job.id] = job

    def _save(self, job: Job) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        path = self.jobs_dir / f"{job.id}.json"
        text = json.dumps(job.public(), indent=2, ensure_ascii=False)
        # 書きかけを読まれないよう、別のファイルに書いてから置き換える。出力を読むスレッドと
        # 止める操作が同時に書くことがあるので、順番に書く
        with self._save_lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            # Windows では、ほかが読んでいる間は置き換えられない。少し待ってやり直す
            for _ in range(40):
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    time.sleep(0.025)
            path.write_text(text, encoding="utf-8")
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ run

    def start(self, workflow: Workflow, args: List[str]) -> Job:
        with self._lock:
            self._counter += 1
            job_id = time.strftime("%Y%m%d_%H%M%S") + f"_{self._counter:02d}"
        job = Job(
            id=job_id,
            workflow=workflow.key,
            title=workflow.title,
            module=workflow.module,
            args=list(args),
            moves_car=workflow.moves_car,
            kind=workflow.kind,
            started_at=_now(),
            produces=produced_path(workflow, args),
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.workdir)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.Popen(
                [self.python, "-m", workflow.module, *args],
                cwd=self.workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            job.status = "failed"
            job.ended_at = _now()
            job.lines.append(f"起動できません: {exc}")
            with self._lock:
                self._jobs[job.id] = job
            self._save(job)
            return job
        with self._lock:
            self._jobs[job.id] = job
            self._procs[job.id] = proc
        self._save(job)
        threading.Thread(target=self._pump, args=(job, proc), daemon=True).start()
        return job

    def _pump(self, job: Job, proc: subprocess.Popen) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.jobs_dir / f"{job.id}.log", "w", encoding="utf-8") as log:
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                log.write(line + "\n")
                log.flush()
                with self._lock:
                    job.lines.append(line)
                    if len(job.lines) > MAX_LINES:
                        drop = len(job.lines) - MAX_LINES
                        del job.lines[:drop]
                        job.line_offset += drop
        code = proc.wait()
        with self._lock:
            job.exit_code = code
            job.ended_at = _now()
            if job.status == "running":
                job.status = "succeeded" if code == 0 else "failed"
            self._procs.pop(job.id, None)
        self._save(job)

    def stop(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            proc = self._procs.get(job_id)
        if proc is None or proc.poll() is not None:
            return job
        notes = []
        if job.moves_car and self.stop_car is not None:
            # 走行は Pi で続くので、プロセスを止める前に車体を止める
            notes.append(self.stop_car())
        job.status = "stopped"
        proc.terminate()
        try:
            proc.wait(timeout=STOP_GRACE_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
        job.stop_note = " / ".join(n for n in notes if n) or None
        self._save(job)
        return job

    def stop_all(self) -> None:
        for job_id in [j.id for j in self.list() if j.status == "running"]:
            self.stop(job_id)

    # ------------------------------------------------------------------ query

    def list(self) -> List[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def lines(self, job_id: str, since: int = 0) -> Dict:
        with self._lock:
            job = self._jobs[job_id]
            if not job.lines and job.status != "running":
                log = self.jobs_dir / f"{job.id}.log"
                if log.is_file():
                    job.lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                    if len(job.lines) > MAX_LINES:
                        job.line_offset = len(job.lines) - MAX_LINES
                        job.lines = job.lines[-MAX_LINES:]
            start = max(0, since - job.line_offset)
            return {
                "lines": job.lines[start:],
                "next": job.line_offset + len(job.lines),
                "status": job.status,
                "exit_code": job.exit_code,
            }

    def running_car_jobs(self) -> List[Job]:
        return [j for j in self.list() if j.status == "running" and j.moves_car]
