use std::time::Instant;
use serde::Deserialize;
use crate::picarx::Picarx;
use crate::sensor_monitor::SensorSnapshot;

// 時間指定コマンドの最小継続時間。LLMが0msなどの退化値を返した場合の安全策
const MIN_DURATION_MS: u64 = 100;

// "eq"/"ne"判定時の許容誤差(センサーの連続値をぴったり一致で比較するのは非現実的なため)
const COMPARE_EPSILON: f64 = 0.5;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorKind {
    Distance,
    GrayscaleLeft,
    GrayscaleCenter,
    GrayscaleRight,
    BatteryVoltage,
    BatteryPercent,
}

impl SensorKind {
    fn read_from(self, s: &SensorSnapshot) -> Option<f64> {
        match self {
            SensorKind::Distance => s.distance_cm,
            SensorKind::GrayscaleLeft => s.grayscale.map(|g| g[0] as f64),
            SensorKind::GrayscaleCenter => s.grayscale.map(|g| g[1] as f64),
            SensorKind::GrayscaleRight => s.grayscale.map(|g| g[2] as f64),
            SensorKind::BatteryVoltage => s.battery.map(|(v, _)| v),
            SensorKind::BatteryPercent => s.battery.map(|(_, p)| p as f64),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operator {
    Eq,
    Ne,
    Gte,
    Lte,
    Gt,
    Lt,
}

impl Operator {
    fn compare(self, lhs: f64, rhs: f64) -> bool {
        match self {
            Operator::Eq => (lhs - rhs).abs() < COMPARE_EPSILON,
            Operator::Ne => (lhs - rhs).abs() >= COMPARE_EPSILON,
            Operator::Gte => lhs >= rhs,
            Operator::Lte => lhs <= rhs,
            Operator::Gt => lhs > rhs,
            Operator::Lt => lhs < rhs,
        }
    }
}

/// 単一センサーの比較条件(条件式の葉ノード)
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Condition {
    pub sensor: SensorKind,
    pub op: Operator,
    pub value: f64,
}

impl Condition {
    fn is_met(&self, s: &SensorSnapshot) -> bool {
        self.sensor.read_from(s).map(|v| self.op.compare(v, self.value)).unwrap_or(false)
    }
}

/// forward/backward/steerに付与できる、センサー条件による早期打ち切り条件。
/// duration_ms経過前でも条件が成立した時点でアクションを終了する。
/// 単一の比較(Condition)に加え、and/or/xor/notで再帰的に組み合わせられる。
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ConditionExpr {
    And { and: Vec<ConditionExpr> },
    Or { or: Vec<ConditionExpr> },
    /// 子要素のうちちょうど1つだけがtrueの場合にtrue
    Xor { xor: Vec<ConditionExpr> },
    Not { not: Box<ConditionExpr> },
    Leaf(Condition),
}

impl ConditionExpr {
    fn is_met(&self, s: &SensorSnapshot) -> bool {
        match self {
            ConditionExpr::And { and } => and.iter().all(|c| c.is_met(s)),
            ConditionExpr::Or { or } => or.iter().any(|c| c.is_met(s)),
            ConditionExpr::Xor { xor } => xor.iter().filter(|c| c.is_met(s)).count() == 1,
            ConditionExpr::Not { not } => !not.is_met(s),
            ConditionExpr::Leaf(cond) => cond.is_met(s),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Command {
    Forward { speed: i32, duration_ms: u64, #[serde(default)] until: Option<ConditionExpr> },
    Backward { speed: i32, duration_ms: u64, #[serde(default)] until: Option<ConditionExpr> },
    Steer { angle: i32, duration_ms: u64, #[serde(default)] until: Option<ConditionExpr> },
    Stop,
    SetCamPan { angle: i32 },
    SetCamTilt { angle: i32 },
    GetDistance,
    GetGrayscale,
    GetBatteryStatus,
}

impl Command {
    fn duration_ms(&self) -> u64 {
        match self {
            Command::Forward { duration_ms, .. }
            | Command::Backward { duration_ms, .. }
            | Command::Steer { duration_ms, .. } => (*duration_ms).max(MIN_DURATION_MS),
            _ => 0,
        }
    }

    fn until(&self) -> Option<&ConditionExpr> {
        match self {
            Command::Forward { until, .. }
            | Command::Backward { until, .. }
            | Command::Steer { until, .. } => until.as_ref(),
            _ => None,
        }
    }
}

/// LLMから受け取ったコマンド列を、メインループから毎フレーム`tick()`することで
/// `thread::sleep`によるブロッキングなしに順次実行するステートマシン。
pub struct CommandQueue {
    commands: Vec<Command>,
    current_index: usize,
    current_started_at: Option<Instant>,
}

impl CommandQueue {
    pub fn new(commands: Vec<Command>) -> Self {
        Self {
            commands,
            current_index: 0,
            current_started_at: None,
        }
    }

    /// 1フレーム分実行を進める。まだ実行中なら true、全コマンド消化済みなら false を返す。
    /// `sensors`はforward/backward/steerに付与された`until`条件の判定に使う
    /// (duration_ms経過前でも、条件が成立した時点で早期に打ち切る)。
    pub fn tick(&mut self, px: &mut Picarx, sensors: SensorSnapshot) -> bool {
        if self.current_index >= self.commands.len() {
            return false;
        }

        if self.current_started_at.is_none() {
            self.start_current(px);
        }

        let current = &self.commands[self.current_index];
        let duration_ms = current.duration_ms();
        let until = current.until();

        let duration_elapsed = self
            .current_started_at
            .map(|t| t.elapsed().as_millis() as u64 >= duration_ms)
            .unwrap_or(true);
        let condition_met = until.map(|c| c.is_met(&sensors)).unwrap_or(false);

        if duration_elapsed || condition_met {
            if condition_met && !duration_elapsed {
                println!("[LLM] センサー条件が成立したため中断します");
            }
            self.finish_current(px);
            self.current_index += 1;
            self.current_started_at = None;
        }

        self.current_index < self.commands.len()
    }

    fn start_current(&mut self, px: &mut Picarx) {
        self.current_started_at = Some(Instant::now());
        match &self.commands[self.current_index] {
            Command::Forward { speed, .. } => px.forward(*speed),
            Command::Backward { speed, .. } => px.backward(*speed),
            Command::Steer { angle, .. } => px.set_dir_servo_angle(*angle),
            Command::Stop => px.stop(),
            Command::SetCamPan { angle } => px.set_cam_pan_angle(*angle),
            Command::SetCamTilt { angle } => px.set_cam_tilt_angle(*angle),
            Command::GetDistance => {
                let distance = px.get_distance();
                if distance < 0.0 {
                    println!("[LLM] Distance: timeout (object too far or hardware error)");
                } else {
                    println!("[LLM] Distance: {:.2} cm", distance);
                }
            }
            Command::GetGrayscale => match px.get_grayscale_data() {
                Ok(values) => println!(
                    "[LLM] Grayscale A0(left): {:4} | A1(center): {:4} | A2(right): {:4}",
                    values[0], values[1], values[2]
                ),
                Err(e) => println!("[LLM] Failed to get grayscale data: {}", e),
            },
            Command::GetBatteryStatus => match px.get_battery_status() {
                Ok((voltage, percent)) => println!("[LLM] Battery: {:.2} V ({}%)", voltage, percent),
                Err(e) => println!("[LLM] Failed to get battery status: {}", e),
            },
        }
    }

    fn finish_current(&mut self, px: &mut Picarx) {
        // タイムベースの移動系コマンドが終わったら安全な状態(停止/直進)に戻す
        match &self.commands[self.current_index] {
            Command::Forward { .. } | Command::Backward { .. } => px.stop(),
            Command::Steer { .. } => px.set_dir_servo_angle(0),
            _ => {}
        }
    }
}
