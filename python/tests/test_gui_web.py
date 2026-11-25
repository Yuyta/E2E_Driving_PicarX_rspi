import http.client
import json
import sys
import threading
import time

import pytest

from src.gui.web import server
from src.gui.web.jobs import JobManager, produced_path
from src.gui.workflows import WORKFLOWS, WORKFLOWS_BY_ID, Workflow


class FakeJobs:
    def __init__(self):
        self.started = []

    def start(self, workflow, args):
        self.started.append((workflow.key, args))

        class Job:
            def public(self_inner):
                return {"id": "job1", "workflow": workflow.key, "args": args}

        return Job()

    def list(self):
        return []

    def running_car_jobs(self):
        return []

    def get(self, job_id):
        return None


class FakeRobot:
    def __init__(self):
        self.calls = []

    def get_status_sync(self):
        raise ConnectionError("offline")

    def emergency_stop_sync(self):
        self.calls.append("estop")
        return True


@pytest.fixture
def running(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "v001").mkdir()
    (tmp_path / "models" / "v001" / "metadata.json").write_text("{}", encoding="utf-8")
    frame = tmp_path / "data" / "raw" / "s1" / "images" / "frame_000003.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"\xff\xd8jpeg")
    robot = FakeRobot()
    app = server.App(base=tmp_path, token="secret", jobs=FakeJobs(), robot=lambda: robot)
    httpd = server.make_server(app, 0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield app, port, robot
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, token="secret", host=None, content_type=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    if token:
        headers["X-PicarX-Token"] = token
    payload = None
    if body is not None:
        payload = json.dumps(body).encode()
        headers["Content-Type"] = content_type or "application/json"
    conn.request(method, path, body=payload, headers=headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res.status, data


def test_index_has_token_and_api_requires_it(running):
    _, port, _ = running
    status, body = request(port, "GET", "/", token=None)
    assert status == 200 and b'content="secret"' in body
    assert request(port, "GET", "/api/catalog", token=None)[0] == 401
    assert request(port, "GET", "/api/catalog", token="wrong")[0] == 401
    assert request(port, "GET", "/api/catalog")[0] == 200


def test_rejects_other_hosts(running):
    _, port, _ = running
    assert request(port, "GET", "/api/catalog", host=f"evil.example:{port}")[0] == 403
    assert request(port, "GET", "/", host="evil.example")[0] == 403


def test_post_must_be_json(running):
    _, port, _ = running
    status, _ = request(
        port, "POST", "/api/jobs", {"workflow": "registry_compare"}, content_type="text/plain"
    )
    assert status == 415


def test_car_moving_tools_need_confirmation(running):
    app, port, _ = running
    status, _ = request(port, "POST", "/api/jobs", {"workflow": "auto_collector"})
    assert status == 409 and app.jobs.started == []
    status, _ = request(port, "POST", "/api/jobs", {"workflow": "deploy_push"})
    assert status == 409 and app.jobs.started == []
    status, _ = request(
        port,
        "POST",
        "/api/jobs",
        {"workflow": "auto_collector", "confirm_moves_car": True, "args": "--seconds 5"},
    )
    assert status == 201
    assert app.jobs.started == [("auto_collector", ["--seconds", "5"])]


def test_unknown_workflow_is_rejected(running):
    _, port, _ = running
    assert request(port, "POST", "/api/jobs", {"workflow": "rm -rf"})[0] == 400


def test_static_files_cannot_escape(running):
    _, port, _ = running
    assert request(port, "GET", "/../server.py", token=None)[0] == 404
    assert request(port, "GET", "/%2e%2e/server.py", token=None)[0] == 404


def test_frames_are_served_only_from_sessions(running):
    _, port, _ = running
    status, body = request(port, "GET", "/api/sessions/s1/frames/3.jpg")
    assert status == 200 and body.startswith(b"\xff\xd8")
    assert request(port, "GET", "/api/sessions/s1/frames/4.jpg")[0] == 404
    assert request(port, "GET", "/api/sessions/..%2F..%2Fmodels/frames/3.jpg")[0] == 404


def test_stage_can_be_changed(running):
    app, port, _ = running
    status, body = request(
        port, "POST", "/api/models/v001/stage", {"stage": "approved", "note": "ok"}
    )
    assert status == 200 and json.loads(body)["stage"] == "approved"
    assert request(port, "POST", "/api/models/v001/stage", {"stage": "bogus"})[0] == 400
    assert request(port, "POST", "/api/models/..%2Fx/stage", {"stage": "approved"})[0] == 400


def test_robot_offline_and_emergency_stop(running):
    _, port, robot = running
    status, body = request(port, "GET", "/api/robot/status")
    assert status == 200 and json.loads(body)["online"] is False
    status, body = request(port, "POST", "/api/robot/emergency_stop", {})
    assert status == 200 and json.loads(body)["ok"] is True
    assert robot.calls == ["estop"]


def test_workflow_ids_are_unique_and_known():
    assert len(WORKFLOWS_BY_ID) == len(WORKFLOWS)
    for workflow in WORKFLOWS:
        if "move" in workflow.description or workflow.module in (
            "src.e2e.pi_autopilot",
            "src.e2e.auto_collector",
            "src.e2e.speed_check",
        ):
            assert workflow.moves_car


def test_produced_path():
    train = WORKFLOWS_BY_ID["train"]
    assert (
        produced_path(train, ["--data_dir", "data/raw", "--output_dir", "models/v009"])
        == "models/v009"
    )
    assert produced_path(train, ["--output_dir=models/v010"]) == "models/v010"
    assert produced_path(WORKFLOWS_BY_ID["m1_check"], ["status"]) is None


# ---------------------------------------------------------------- JobManager (real processes)


def wait_until(predicate, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_job_runs_and_is_saved(tmp_path):
    (tmp_path / "hello_mod.py").write_text("print('こんにちは')\nprint('done')\n", encoding="utf-8")
    jobs = JobManager(tmp_path, tmp_path / "jobs", python=sys.executable)
    job = jobs.start(Workflow("hello", "", "hello_mod", "g", id="hello"), [])
    assert wait_until(lambda: jobs.get(job.id).status != "running")
    assert jobs.get(job.id).status == "succeeded"
    assert jobs.lines(job.id)["lines"] == ["こんにちは", "done"]
    saved = json.loads((tmp_path / "jobs" / f"{job.id}.json").read_text(encoding="utf-8"))
    assert saved["exit_code"] == 0 and saved["command"] == "python -m hello_mod"
    # 開き直しても履歴とログが読める
    again = JobManager(tmp_path, tmp_path / "jobs", python=sys.executable)
    assert again.lines(job.id)["lines"] == ["こんにちは", "done"]


def test_stopping_a_car_job_stops_the_car_first(tmp_path):
    (tmp_path / "sleeper.py").write_text(
        "import time\nprint('start', flush=True)\ntime.sleep(60)\n", encoding="utf-8"
    )
    order = []
    jobs = JobManager(
        tmp_path, tmp_path / "jobs", stop_car=lambda: order.append("car") or "car stopped"
    )
    job = jobs.start(Workflow("drive", "", "sleeper", "g", moves_car=True, id="drive"), [])
    assert wait_until(lambda: jobs.lines(job.id)["lines"] == ["start"])
    stopped = jobs.stop(job.id)
    order.append("process")
    assert order == ["car", "process"]
    assert stopped.status == "stopped" and stopped.stop_note == "car stopped"
    assert wait_until(lambda: jobs.get(job.id).exit_code is not None)
    assert jobs.get(job.id).status == "stopped"


def test_every_tool_has_a_manual_section(running):
    """ツールの「使い方」のリンク先が、マニュアルに必ずある"""
    _, port, _ = running
    status, body = request(port, "GET", "/manual.html", token=None)
    assert status == 200
    html = body.decode("utf-8")
    windows = {"app": "gui-app", "collector": "gui-collector", "data_viewer": "gui-viewer"}
    missing = [
        w.key for w in WORKFLOWS if f'id="{windows.get(w.key, "tool-" + w.key)}"' not in html
    ]
    assert missing == []


def test_phases_and_workflows_share_the_phase_ids(running):
    _, port, _ = running
    status, body = request(port, "GET", "/api/phases")
    assert status == 200
    phases = {p["id"]: p for p in json.loads(body)}
    assert phases["train"]["number"] == 3 and phases["train"]["page"] == "models"
    status, body = request(port, "GET", "/api/workflows")
    assert all(w["phase"] in phases for w in json.loads(body))


# ---------------------------------------------------------------- folder and file arguments


def test_browse_lists_only_inside_the_project(tmp_path):
    (tmp_path / "data" / "raw" / "s1").mkdir(parents=True)
    (tmp_path / "data" / "cv.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data" / ".hidden").mkdir()
    (tmp_path / "data" / "__pycache__").mkdir()
    app = server.App(base=tmp_path, token="t", jobs=object(), robot=lambda: None)
    listing = app.browse("data")
    assert listing["path"] == "data" and listing["dirs"] == ["raw"]
    assert [f["name"] for f in listing["files"]] == ["cv.json"]
    # まだない出力先は、あるところまで上がって開く
    assert app.browse("data/cv/new_cv.json")["path"] == "data"
    assert app.browse("")["path"] == ""
    with pytest.raises(PermissionError):
        app.browse("..")
    with pytest.raises(PermissionError):
        app.browse("data/../../outside")


def test_browse_and_path_args_over_http(running):
    _, port, _ = running
    status, body = request(port, "GET", "/api/path_args")
    assert status == 200 and json.loads(body)["--data_dir"]["kind"] == "dir"
    status, _ = request(port, "GET", "/api/browse?path=..")
    assert status == 409
    status, _ = request(port, "GET", "/api/browse?path=.", token=None)
    assert status == 401


def test_path_arguments_split_and_join_back():
    from src.gui.workflows import PATH_ARGS, join_path_args, split_path_args

    tokens = ["push", "--model_dir", "models/v006", "--output=dist/models", "--verify"]
    rest, fields = split_path_args(tokens)
    assert rest == ["push", "--verify"]
    assert fields == [("--model_dir", "models/v006"), ("--output", "dist/models")]
    assert join_path_args(rest, fields) == [
        "push",
        "--verify",
        "--model_dir",
        "models/v006",
        "--output",
        "dist/models",
    ]
    # 既定の引数にあるフォルダやファイルの引数は、すべて欄になる
    for workflow in WORKFLOWS:
        _, found = split_path_args(workflow.arguments.split())
        flags = [
            t
            for t in workflow.arguments.split()
            if t.startswith("--") and "/" in workflow.arguments
        ]
        assert all(f in PATH_ARGS for f, _ in found)
        assert not any(t in PATH_ARGS for t in flags) or found


def test_components_page_lists_parts_and_versions(running):
    """構成の画面: 部品の一覧と、部品ごとの版。トークンなしでは読めない"""
    _, port, _ = running
    assert request(port, "GET", "/api/components", token=None)[0] == 401
    status, body = request(port, "GET", "/api/components")
    assert status == 200
    data = json.loads(body)
    ids = {c["id"] for c in data["components"]}
    assert {"reflex", "autopilot", "agent", "mlops"} <= ids
    assert set(data["versions"]) == {"pi_server", "model", "pc_code", "body"}
    # 走行の記録がまだない時は Pi の版は分からない
    assert data["versions"]["pi_server"]["value"] is None
    status, body = request(port, "GET", "/", token=None)
    assert b'data-route="components"' in body


def test_components_versions_come_from_the_records():
    from src.gui import components

    summary = {
        "last_pi_run": {
            "model_version": "v005",
            "server_build": "picar-x-rust 0.1.0 (6c139ea)",
            "started_at": "2026-11-16T21:58:58+0900",
        },
        "deployed": {"version": "v006", "at": "2026-11-24T21:20:07+0900"},
        "body_current": "b3",
    }
    v = components.versions(summary, {"commit": "0123456789abcdef", "dirty": True})
    assert v["pi_server"]["value"] == "picar-x-rust 0.1.0 (6c139ea)"
    assert v["model"] == {
        "value": "v006",
        "at": "2026-11-24T21:20:07+0900",
        "note": "デプロイの記録から",
    }
    assert v["pc_code"]["value"] == "0123456" and v["pc_code"]["dirty"]
    assert v["body"]["value"] == "b3"
    # デプロイの記録がなければ、最後の走行のモデル
    summary["deployed"] = {"version": None}
    assert components.versions(summary, {})["model"]["value"] == "v005"
    assert components.versions({}, {})["model"]["note"] == "記録なし"


def test_only_the_pi_holds_the_safety_core():
    """安全の要は Pi の部品だけ。PC の部品は安全から切り離す"""
    from src.gui.components import COMPONENTS, Component

    assert all(c.place == "pi" for c in COMPONENTS if c.safety == "core")
    assert all(c.safety == "isolated" for c in COMPONENTS if c.place == "pc")
    assert len({c.id for c in COMPONENTS}) == len(COMPONENTS)
    with pytest.raises(ValueError):
        Component("x", "x", "cloud", "", "core", "", "", "", "pi_server")
