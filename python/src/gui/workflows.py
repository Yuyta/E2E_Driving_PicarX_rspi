"""ランチャーと Control Center が起動するツールの一覧と、MLOps のフェーズ

各ツールは既存のモジュール（`python -m <module> <引数>`）をそのまま別プロセスで動かす。
ここでは画面に出す名前、既定の引数、確認の要否、どのフェーズで使うかだけを決め、
ツールの中身は変えない。

フェーズは「車体準備 → データ収集 → データ整備 → モデル学習 → モデル評価 → デプロイ
→ 実走行 → データ収集」の輪。
画面の色とアイコンはここで決めた値を、ランチャー（Qt）と Control Center（ブラウザ）の両方で使う。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Phase:
    id: str
    # 輪の中の順番。共通（いつでも使う）は None
    number: Optional[int]
    title: str
    summary: str
    color: str
    # Control Center でこのフェーズの中身を見る画面（#/<page>）
    page: str

    @property
    def icon(self) -> str:
        """24x24 の線のアイコン（SVG の中身）。ランチャーと Control Center で同じ絵を使う"""
        return PHASE_ICONS[self.id]


# 番号ではなく絵でフェーズを見分ける。stroke は呼ぶ側で currentColor などを指定する
PHASE_ICONS = {
    # 準備: レンチ（車体の点検と校正）
    "prepare": '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3.5 17.5l3 3 5.8-5.8a4 4 0 0 0 5.4-5.4'
    'l-2.6 2.6-2.3-.6-.6-2.3z"/>',
    # 収集: カメラ（走行データの記録）
    "collect": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    # 整備: じょうご（確かめて、除いて、封をする）
    "curate": '<path d="M3.5 5h17l-6.5 7.5V19l-4-2v-4.5z"/>',
    # 学習: つながった節（モデル）
    "train": '<circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/>'
    '<circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/>'
    '<path d="M6.6 7.3l3.8 3.4M6.6 16.7l3.8-3.4M14 12h3"/>',
    # 評価: 計器（比べて採否を決める）
    "evaluate": '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4.2-5.4"/>'
    '<path d="M6.5 11.5l1 .6M12 7v1.2M17.5 11.5l-1 .6"/>',
    # デプロイ: 上向きの矢印（Pi へ送る）
    "deploy": '<path d="M12 15V4M7.5 8.5 12 4l4.5 4.5"/><path d="M5 14v5h14v-5"/>',
    # 実走行: ハンドル（走らせる）
    "operate": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="2"/>'
    '<path d="M3.6 11h6.4M14 11h6.4M12 14v6.4"/>',
    # 共通: 重なった層
    "common": '<path d="M12 3.5 20.5 8 12 12.5 3.5 8z"/><path d="M3.5 12.5 12 17l8.5-4.5"/>',
}


PHASES = (
    Phase("prepare", 0, "車体準備", "車体を点検し、装備と校正を記録する", "#6f7a8c", "body"),
    Phase("collect", 1, "データ収集", "走らせて走行データを記録する", "#2a78d6", "data"),
    Phase("curate", 2, "データ整備", "確かめて、ラベルを付け、封をする", "#13899a", "data"),
    Phase("train", 3, "モデル学習", "走行データからモデルを作る", "#7b5fd6", "models"),
    Phase("evaluate", 4, "モデル評価", "比べて、使うモデルを決める", "#c46f1a", "compare"),
    Phase("deploy", 5, "デプロイ", "モデルを Pi へ送って使用中にする", "#2f9e5b", "deploy"),
    Phase("operate", 6, "実走行", "Pi で走らせ、走行と介入を記録する", "#c8467f", "runs"),
    Phase("common", None, "共通", "どの段階でも使う", "#898781", "tools"),
)
PHASES_BY_ID = {phase.id: phase for phase in PHASES}


@dataclass(frozen=True)
class Workflow:
    title: str
    description: str
    module: str
    # 使うフェーズ（PHASES の id）
    phase: str
    moves_car: bool = False
    arguments: str = ""
    examples: Tuple[Tuple[str, str], ...] = ()
    # 起動前に確かめる内容（車体は動かないが、Pi のモデルを変えるなど）
    confirm: Optional[str] = None
    # window: 別の画面を開く。job: 出力をログに流して終わる
    kind: str = "job"
    # この引数の値が、作られる成果物（例: --output_dir models/v007）
    produces: Optional[str] = None
    id: str = field(default="")

    @property
    def key(self) -> str:
        return self.id or self.module


WORKFLOWS = (
    # ---------------------------------------------------------------- 0 車体準備
    Workflow(
        "安全チェック",
        "M1 の車体・安全機能を確認",
        "src.e2e.m1_check",
        "prepare",
        arguments="status",
        examples=(
            ("状態", "status"),
            ("カメラ計測", "camera --seconds 20"),
            ("安全イベント", "events"),
        ),
        id="m1_check",
    ),
    Workflow(
        "動きの計測",
        "M3 の動き計測・校正",
        "src.e2e.m3_check",
        "prepare",
        arguments="run --scenario still",
        examples=(
            ("静止", "run --scenario still"),
            ("直進（車体が動く）", "run --scenario forward --i-understand-the-car-will-move"),
            ("校正", "calibrate"),
        ),
        id="m3_check",
    ),
    Workflow(
        "速度チェック",
        "車輪が回り始める速度を計測",
        "src.e2e.speed_check",
        "prepare",
        True,
        "--i-understand-the-car-will-move",
        (("速度確認", "--speeds 30 35 40 45 --i-understand-the-car-will-move"),),
        id="speed_check",
    ),
    Workflow(
        "車体の版の一覧",
        "登録した車体の版（装備と校正）と、今使っている版を表示",
        "src.e2e.registry.body",
        "prepare",
        arguments="show",
        id="body_show",
    ),
    Workflow(
        "車体の版を登録",
        "装備や校正を変えた時に、新しい車体の版を configs/body.toml に足す",
        "src.e2e.registry.body",
        "prepare",
        arguments='add --id b4 --title "M6 の後" --equipment "ジャイロ（I2C）" '
        "--set motor_balance=0",
        examples=(
            (
                "M6 の後（ジャイロを追加）",
                'add --id b4 --title "M6 の後" --equipment "前向きカメラ" '
                '--equipment "前方の超音波 1 本" --equipment "床のグレースケール 3ch" '
                '--equipment "ジャイロ（I2C）" --set dir_servo_offset=6 --set motor_balance=0',
            ),
        ),
        confirm="configs/body.toml に新しい車体の版を足します。"
        "これ以降に集めたデータは、この版の車体のデータとして扱われます。",
        id="body_add",
    ),
    # ---------------------------------------------------------------- 1 データ収集
    Workflow(
        "走行データ収集",
        "カメラを見ながら操作と走行データを記録",
        "src.e2e.gui_collector",
        "collect",
        kind="window",
        id="collector",
    ),
    Workflow(
        "自動データ収集",
        "ルールに従って車体を走らせデータを記録",
        "src.e2e.auto_collector",
        "collect",
        True,
        "--i-understand-the-car-will-move",
        (("150秒（車体が動く）", "--seconds 150 --i-understand-the-car-will-move"),),
        id="auto_collector",
    ),
    # ---------------------------------------------------------------- 2 データ整備
    Workflow(
        "データ確認・ラベル",
        "走行データの確認、注釈、統計",
        "src.e2e.data_viewer",
        "curate",
        kind="window",
        id="data_viewer",
    ),
    Workflow(
        "マニフェスト作成・検証",
        "セッションのファイルのハッシュを記録し、変わっていないか確かめる",
        "src.e2e.dataset.manifest",
        "curate",
        arguments="--data_dir data/raw --verify",
        examples=(
            ("検証", "--data_dir data/raw --verify"),
            ("未作成のセッションに作る", "--data_dir data/raw"),
        ),
        id="manifest",
    ),
    Workflow(
        "セッションの報告",
        "セッションごとの記録の質（介入・食い違い・停止）を集計",
        "src.e2e.dataset.session_report",
        "curate",
        arguments="--data_dir data/raw",
        id="session_report",
    ),
    Workflow(
        "データのクリーニング",
        "モデルの誤差が大きいフレームを外したデータを作る",
        "src.e2e.dataset.cleaner",
        "curate",
        arguments="--data_dir data/raw --model models/v006/model.onnx "
        "--output_dir data/processed/clean_v1",
        produces="--output_dir",
        id="cleaner",
    ),
    # ---------------------------------------------------------------- 3 モデル学習
    Workflow(
        "モデル学習",
        "記録データからモデルを学習し ONNX を出力",
        "src.e2e.training.train",
        "train",
        arguments="--config configs/train.toml --data_dir data/raw --output_dir models/v007",
        produces="--output_dir",
        id="train",
    ),
    # ---------------------------------------------------------------- 4 モデル評価
    Workflow(
        "交差検証",
        "セッションを分けてモデルを評価",
        "src.e2e.training.cross_validate",
        "evaluate",
        arguments="--config configs/train.toml --data_dir data/raw --output data/cv/new_cv.json",
        produces="--output",
        id="cross_validate",
    ),
    Workflow(
        "交差検証の比較",
        "2つの交差検証の結果を比べ、採用してよいかを判定",
        "src.e2e.evaluation.compare_cv",
        "evaluate",
        arguments="--current data/cv/auto_8sessions_turn.json "
        "--candidate data/cv/new_cv.json --output data/cv/compare_new.json",
        produces="--output",
        id="compare_cv",
    ),
    Workflow(
        "オフライン評価",
        "モデルを記録データで評価し eval_metrics.json を書く",
        "src.e2e.evaluation.evaluate",
        "evaluate",
        arguments="--data_dir data/raw --model_path models/v006/best_model.pth",
        id="evaluate",
    ),
    Workflow(
        "門番の試験",
        "門番のタグを付けた場面でモデルを1フレームずつ採点し、gate.json に残す"
        "（合格しないモデルは Pi へ送らない）",
        "src.e2e.evaluation.gate",
        "evaluate",
        arguments="--model_dir models/v006",
        examples=(("採点", "--model_dir models/v006"), ("場面の一覧", "--list")),
        id="gate",
    ),
    Workflow(
        "モデル比較",
        "登録済みモデルの評価値を比較",
        "src.e2e.registry.compare",
        "evaluate",
        id="registry_compare",
    ),
    # ---------------------------------------------------------------- 5 デプロイ
    Workflow(
        "バンドル作成",
        "Pi へ送るモデルのフォルダを作り、ハッシュと入力の大きさを確かめる",
        "src.e2e.deploy.bundle",
        "deploy",
        arguments="--model_dir models/v006 --output dist/models",
        id="bundle",
    ),
    Workflow(
        "Pi へデプロイ",
        "門番の試験に合格したモデルを Pi へ送り、確かめてから使用中にする",
        "src.e2e.deploy.pi",
        "deploy",
        arguments="push --model_dir models/v006",
        confirm="Pi で使うモデルが変わります。次の自律走行の開始から使われます。",
        id="deploy_push",
    ),
    Workflow(
        "Pi のモデルを戻す",
        "Pi で1つ前に使っていた版へ戻す",
        "src.e2e.deploy.pi",
        "deploy",
        arguments="rollback",
        confirm="Pi で使うモデルが1つ前の版に戻ります。",
        id="deploy_rollback",
    ),
    Workflow(
        "Pi のモデル一覧",
        "Pi に入っている版と使用中の版を表示",
        "src.e2e.deploy.pi",
        "deploy",
        arguments="list",
        id="deploy_list",
    ),
    # ---------------------------------------------------------------- 6 実走行
    Workflow(
        "Pi 自律走行",
        "選択したモデルで Pi の車体を自律走行",
        "src.e2e.pi_autopilot",
        "operate",
        True,
        "--speed 50",
        (("意図: 左", "--speed 50 --intent left"), ("パッドで介入", "--speed 50 --gamepad")),
        id="pi_autopilot",
    ),
    Workflow(
        "走行記録の取り込み",
        "Pi の走行の記録を data/runs へ持ってくる",
        "src.e2e.deploy.pi",
        "operate",
        arguments="pull-runs",
        id="deploy_pull_runs",
    ),
    Workflow(
        "介入率の集計",
        "走行の記録から、モデル別の介入の回数を集計",
        "src.e2e.evaluation.interventions",
        "operate",
        arguments="--runs_dir data/runs",
        id="interventions",
    ),
    # ---------------------------------------------------------------- 共通
    Workflow(
        "会話・手動操作",
        "車体の状態確認、会話、手動操作",
        "src.gui.app",
        "common",
        kind="window",
        id="app",
    ),
    # Pi の走行サーバー（ssh で起動・停止。車は動かない）
    Workflow(
        "Pi の走行サーバーを起動",
        "ssh で Pi へ入り、走行サーバー（picar-x-rust）を起動する。動いていれば何もしない",
        "src.e2e.deploy.pi_server",
        "common",
        arguments="start",
        id="pi_server_start",
    ),
    Workflow(
        "Pi の走行サーバーを起動（床用）",
        "段差の検知を切って走行サーバーを起動する。黒いマットで止まらないよう、床で走らせる時だけ使う",
        "src.e2e.deploy.pi_server",
        "common",
        arguments="start --no_cliff_guard",
        confirm="段差の検知を切ります。机の上など、落ちる所では走らせないでください。",
        id="pi_server_start_floor",
    ),
    Workflow(
        "Pi の走行サーバーを停止",
        "走行サーバーを止める。自律走行中なら止めてから終わる",
        "src.e2e.deploy.pi_server",
        "common",
        arguments="stop",
        confirm="Pi の走行サーバーを止めます。止めている間は、会話や手動操作で車を動かせません。",
        id="pi_server_stop",
    ),
    Workflow(
        "Pi の走行サーバーの状態",
        "動いているか、段差の検知が有効か、出力の最後の行",
        "src.e2e.deploy.pi_server",
        "common",
        arguments="status",
        examples=(("状態", "status"), ("出力の記録", "log --lines 50")),
        id="pi_server_status",
    ),
)

WORKFLOWS_BY_ID = {workflow.key: workflow for workflow in WORKFLOWS}


@dataclass(frozen=True)
class PathArg:
    """フォルダやファイルを指す引数。起動の画面で、欄と「選ぶ…」のボタンにする

    kind: dir（あるフォルダ）/ file（あるファイル）/ new（これから作るフォルダかファイル。
    置き場所のフォルダを選び、名前は欄で決める）
    """

    label: str
    kind: str
    exts: Tuple[str, ...] = ()


PATH_ARGS = {
    "--config": PathArg("設定ファイル", "file", (".toml",)),
    "--data_dir": PathArg("データのフォルダ", "dir"),
    "--model_dir": PathArg("モデルのフォルダ", "dir"),
    "--models_dir": PathArg("モデル置き場", "dir"),
    "--runs_dir": PathArg("走行記録のフォルダ", "dir"),
    "--model": PathArg("モデル", "file", (".onnx", ".pth")),
    "--model_path": PathArg("モデルの重み", "file", (".pth",)),
    "--current": PathArg("今の交差検証の結果", "file", (".json",)),
    "--candidate": PathArg("候補の交差検証の結果", "file", (".json",)),
    "--output_dir": PathArg("出力先のフォルダ", "new"),
    "--output": PathArg("結果の保存先", "new"),
}


def workflows_in(phase_id: str) -> Tuple[Workflow, ...]:
    return tuple(w for w in WORKFLOWS if w.phase == phase_id)


def split_path_args(tokens):
    """引数の並びを、フォルダやファイルの引数（PATH_ARGS）とそれ以外に分ける"""
    rest, fields = [], []
    i = 0
    while i < len(tokens):
        flag, eq, inline = tokens[i].partition("=")
        if flag in PATH_ARGS:
            if eq:
                value = inline
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 1
                value = tokens[i]
            else:
                value = ""
            fields.append((flag, value))
        else:
            rest.append(tokens[i])
        i += 1
    return rest, fields


def join_path_args(rest, fields):
    """split_path_args の逆。位置で決まる引数（push など）が先に来るよう、ほかの引数を前に置く"""
    out = list(rest)
    for flag, value in fields:
        if value.strip():
            out += [flag, value.strip()]
    return out
