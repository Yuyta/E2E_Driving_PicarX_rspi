use std::time::Instant;
use serde::Deserialize;
use crate::picarx::Picarx;

// 時間指定コマンドの最小継続時間。LLMが0msなどの退化値を返した場合の安全策
const MIN_DURATION_MS: u64 = 100;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Command {
    Forward { speed: i32, duration_ms: u64 },
    Backward { speed: i32, duration_ms: u64 },
    Steer { angle: i32, duration_ms: u64 },
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
    pub fn tick(&mut self, px: &mut Picarx) -> bool {
        if self.current_index >= self.commands.len() {
            return false;
        }

        if self.current_started_at.is_none() {
            self.start_current(px);
        }

        let duration_ms = self.commands[self.current_index].duration_ms();
        let elapsed_ok = self
            .current_started_at
            .map(|t| t.elapsed().as_millis() as u64 >= duration_ms)
            .unwrap_or(true);

        if elapsed_ok {
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
