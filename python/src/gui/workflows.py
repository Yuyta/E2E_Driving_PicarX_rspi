"""ランチャーと Control Center が起動するツールの一覧と、MLOps のフェーズ

各ツールは既存のモジュール（`python -m <module> <引数>`）をそのまま別プロセスで動かす。
ここでは画面に出す名前、既定の引数、確認の要否、どのフェーズで使うかだけを決め、
ツールの中身は変えない。

フェーズは「準備 → 収集 → 整備 → 学習 → 評価 → 配置 → 運用 → 次の収集」の輪。
画面の色と番号はここで決めた値を、ランチャー（Qt）と Control Center（ブラウザ）の両方で使う。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Phase:
    id: str
    # 輪の中の順番。横断（いつでも使う）は None
    number: Optional[int]
    title: str
    summary: str
    color: str
    # Control Center でこのフェーズの中身を見る画面（#/<page>）
    page: str


PHASES = (
    Phase(
        "prepare", 0, "準備", "車体を確かめ、車体の版（装備と校正）を記録する", "#6f7a8c", "body"
    ),
    Phase("collect", 1, "収集", "車体を走らせて走行データを記録する", "#2a78d6", "data"),
    Phase(
        "curate",
        2,
        "整備",
        "データを確かめ、ラベルを付け、変わっていないか検証する",
        "#13899a",
        "data",
    ),
    Phase("train", 3, "学習", "記録データからモデルを作る", "#7b5fd6", "models"),
    Phase(
        "evaluate", 4, "評価", "モデルを比べ、使うかどうか（採否）を決める", "#c46f1a", "compare"
    ),
    Phase("deploy", 5, "配置", "モデルを Pi へ送り、使用中にする", "#2f9e5b", "deploy"),
    Phase("operate", 6, "運用", "Pi で走らせ、走行と介入の記録を集める", "#c8467f", "runs"),
    Phase("common", None, "横断", "どのフェーズでも使う", "#898781", "tools"),
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
    # ---------------------------------------------------------------- 0 準備
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
    # ---------------------------------------------------------------- 1 収集
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
    # ---------------------------------------------------------------- 2 整備
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
    # ---------------------------------------------------------------- 3 学習
    Workflow(
        "モデル学習",
        "記録データからモデルを学習し ONNX を出力",
        "src.e2e.training.train",
        "train",
        arguments="--config configs/train.toml --data_dir data/raw --output_dir models/v007",
        produces="--output_dir",
        id="train",
    ),
    # ---------------------------------------------------------------- 4 評価
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
        "モデル比較",
        "登録済みモデルの評価値を比較",
        "src.e2e.registry.compare",
        "evaluate",
        id="registry_compare",
    ),
    # ---------------------------------------------------------------- 5 配置
    Workflow(
        "バンドル作成",
        "Pi へ送るモデルのフォルダを作り、ハッシュと入力の大きさを確かめる",
        "src.e2e.deploy.bundle",
        "deploy",
        arguments="--model_dir models/v006 --output dist/models",
        id="bundle",
    ),
    Workflow(
        "Pi へ配置",
        "モデルを Pi へ送り、確かめてから使用中にする",
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
    # ---------------------------------------------------------------- 6 運用
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
    # ---------------------------------------------------------------- 横断
    Workflow(
        "会話・手動操作",
        "車体の状態確認、会話、手動操作",
        "src.gui.app",
        "common",
        kind="window",
        id="app",
    ),
)

WORKFLOWS_BY_ID = {workflow.key: workflow for workflow in WORKFLOWS}


def workflows_in(phase_id: str) -> Tuple[Workflow, ...]:
    return tuple(w for w in WORKFLOWS if w.phase == phase_id)
