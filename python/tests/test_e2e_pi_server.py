"""Pi の走行サーバーを PC から起動、停止、確認する（ssh で送る sh の中身と、接続先）"""

from types import SimpleNamespace

import pytest

from src.e2e.deploy import pi, pi_server


class Recorder:
    def __init__(self, code=0):
        self.calls = []
        self.code = code

    def __call__(self, cmd, **_kwargs):
        self.calls.append(cmd)
        return SimpleNamespace(returncode=self.code)


@pytest.fixture(autouse=True)
def no_dotenv(tmp_path, monkeypatch):
    """利用者の python/.env の接続先を読まない"""
    monkeypatch.setattr(pi, "DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.delenv(pi.HOST_ENV, raising=False)


def test_remote_paths_expand_the_home_directory_outside_quotes():
    assert pi_server.remote_path("~/picar-x-rust") == '"$HOME"/picar-x-rust'
    assert pi_server.remote_path("~/a b") == "\"$HOME\"/'a b'"
    assert pi_server.remote_path("/opt/picar-x-rust") == "/opt/picar-x-rust"


def test_start_runs_in_the_background_next_to_its_env_file():
    script = pi_server.start_script("~/picar-x-rust", no_cliff_guard=False)
    assert "NAME=picar-x-rust" in script
    assert 'cd "$(dirname "$BIN")"' in script  # .env を同じフォルダから読む
    assert 'setsid nohup "$BIN" > "$LOG" 2>&1 < /dev/null &' in script
    assert "PICARX_DISABLE_CLIFF_GUARD=1 setsid" not in script
    assert "もう動いています" in script  # 二重に起動しない
    floor = pi_server.start_script("~/picar-x-rust", no_cliff_guard=True)
    assert "PICARX_DISABLE_CLIFF_GUARD=1 setsid nohup" in floor


def test_stop_uses_sigterm_so_the_server_stops_the_autopilot_first():
    script = pi_server.stop_script("~/picar-x-rust")
    assert 'pkill -TERM -x "$NAME"' in script
    assert "-KILL" not in script


def test_ssh_never_waits_for_a_password(monkeypatch):
    run = Recorder()
    monkeypatch.setattr(pi_server.subprocess, "run", run)
    assert pi_server.main(["--host", "rspi", "status"]) == 0
    cmd = run.calls[0]
    assert cmd[0] == "ssh" and "BatchMode=yes" in cmd and cmd[-2] == "rspi"
    assert "pgrep" in cmd[-1]


def test_host_can_come_from_python_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("PICARX_SSH_HOST=rspi\n", encoding="utf-8")
    monkeypatch.setattr(pi, "DOTENV_PATH", env)
    run = Recorder()
    monkeypatch.setattr(pi_server.subprocess, "run", run)
    assert pi_server.main(["stop"]) == 0
    assert run.calls[0][-2] == "rspi"


def test_a_missing_host_is_an_error():
    with pytest.raises(SystemExit):
        pi_server.main(["status"])


def test_ssh_failure_explains_the_key_setup(monkeypatch, capsys):
    monkeypatch.setattr(pi_server.subprocess, "run", Recorder(code=255))
    assert pi_server.main(["--host", "rspi", "start"]) == 255
    assert "鍵でパスワードなしに入れるか" in capsys.readouterr().err
