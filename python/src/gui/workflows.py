"""ランチャーと Control Center が起動するツールの一覧

各ツールは既存のモジュール（`python -m <module> <引数>`）をそのまま別プロセスで動かす。
ここでは画面に出す名前、既定の引数、確認の要否だけを決め、ツールの中身は変えない。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Workflow:
    title: str
    description: str
    module: str
    group: str
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
    Workflow(
        "会話・手動操作",
        "車体の状態確認、会話、手動操作",
        "src.gui.app",
        "操作",
        kind="window",
        id="app",
    ),
    Workflow(
        "走行データ収集",
        "カメラを見ながら操作と走行データを記録",
        "src.e2e.gui_collector",
        "データ",
        kind="window",
        id="collector",
    ),
    Workflow(
        "データ確認・ラベル",
        "走行データの確認、注釈、統計",
        "src.e2e.data_viewer",
        "データ",
        kind="window",
        id="data_viewer",
    ),
    Workflow(
        "マニフェスト作成・検証",
        "セッションのファイルのハッシュを記録し、変わっていないか確かめる",
        "src.e2e.dataset.manifest",
        "データ",
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
        "データ",
        arguments="--data_dir data/raw",
        id="session_report",
    ),
    Workflow(
        "データのクリーニング",
        "モデルの誤差が大きいフレームを外したデータを作る",
        "src.e2e.dataset.cleaner",
        "データ",
        arguments="--data_dir data/raw --model models/v006/model.onnx "
        "--output_dir data/processed/clean_v1",
        produces="--output_dir",
        id="cleaner",
    ),
    Workflow(
        "モデル学習",
        "記録データからモデルを学習し ONNX を出力",
        "src.e2e.training.train",
        "学習・評価",
        arguments="--config configs/train.toml --data_dir data/raw --output_dir models/v007",
        produces="--output_dir",
        id="train",
    ),
    Workflow(
        "交差検証",
        "セッションを分けてモデルを評価",
        "src.e2e.training.cross_validate",
        "学習・評価",
        arguments="--config configs/train.toml --data_dir data/raw --output data/cv/new_cv.json",
        produces="--output",
        id="cross_validate",
    ),
    Workflow(
        "交差検証の比較",
        "2つの交差検証の結果を比べ、採用してよいかを判定",
        "src.e2e.evaluation.compare_cv",
        "学習・評価",
        arguments="--current data/cv/auto_8sessions_turn.json "
        "--candidate data/cv/new_cv.json --output data/cv/compare_new.json",
        produces="--output",
        id="compare_cv",
    ),
    Workflow(
        "オフライン評価",
        "モデルを記録データで評価し eval_metrics.json を書く",
        "src.e2e.evaluation.evaluate",
        "学習・評価",
        arguments="--data_dir data/raw --model_path models/v006/best_model.pth",
        id="evaluate",
    ),
    Workflow(
        "モデル比較",
        "登録済みモデルの評価値を比較",
        "src.e2e.registry.compare",
        "学習・評価",
        id="registry_compare",
    ),
    Workflow(
        "介入率の集計",
        "走行の記録から、モデル別の介入の回数を集計",
        "src.e2e.evaluation.interventions",
        "学習・評価",
        arguments="--runs_dir data/runs",
        id="interventions",
    ),
    Workflow(
        "バンドル作成",
        "Pi へ送るモデルのフォルダを作り、ハッシュと入力の大きさを確かめる",
        "src.e2e.deploy.bundle",
        "配置",
        arguments="--model_dir models/v006 --output dist/models",
        id="bundle",
    ),
    Workflow(
        "Pi へ配置",
        "モデルを Pi へ送り、確かめてから使用中にする",
        "src.e2e.deploy.pi",
        "配置",
        arguments="push --model_dir models/v006",
        confirm="Pi で使うモデルが変わります。次の自律走行の開始から使われます。",
        id="deploy_push",
    ),
    Workflow(
        "Pi のモデルを戻す",
        "Pi で1つ前に使っていた版へ戻す",
        "src.e2e.deploy.pi",
        "配置",
        arguments="rollback",
        confirm="Pi で使うモデルが1つ前の版に戻ります。",
        id="deploy_rollback",
    ),
    Workflow(
        "Pi のモデル一覧",
        "Pi に入っている版と使用中の版を表示",
        "src.e2e.deploy.pi",
        "配置",
        arguments="list",
        id="deploy_list",
    ),
    Workflow(
        "走行記録の取り込み",
        "Pi の走行の記録を data/runs へ持ってくる",
        "src.e2e.deploy.pi",
        "配置",
        arguments="pull-runs",
        id="deploy_pull_runs",
    ),
    Workflow(
        "安全チェック",
        "M1 の車体・安全機能を確認",
        "src.e2e.m1_check",
        "車体チェック",
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
        "車体チェック",
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
        "車体チェック",
        True,
        "--i-understand-the-car-will-move",
        (("速度確認", "--speeds 30 35 40 45 --i-understand-the-car-will-move"),),
        id="speed_check",
    ),
    Workflow(
        "Pi 自律走行",
        "選択したモデルで Pi の車体を自律走行",
        "src.e2e.pi_autopilot",
        "走行",
        True,
        "--speed 50",
        (("意図: 左", "--speed 50 --intent left"), ("パッドで介入", "--speed 50 --gamepad")),
        id="pi_autopilot",
    ),
    Workflow(
        "自動データ収集",
        "ルールに従って車体を走らせデータを記録",
        "src.e2e.auto_collector",
        "走行",
        True,
        "--i-understand-the-car-will-move",
        (("150秒（車体が動く）", "--seconds 150 --i-understand-the-car-will-move"),),
        id="auto_collector",
    ),
)

WORKFLOWS_BY_ID = {workflow.key: workflow for workflow in WORKFLOWS}
