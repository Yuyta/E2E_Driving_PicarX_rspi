use crate::picarx::Picarx;
use eframe::egui;
use std::collections::HashMap;

// 走行・ステアリング
const MOVE_SPEED: i32 = 80;
const STEER_ANGLE: i32 = 35;

// カメラPan/Tilt (1キー入力あたりの刻み幅と可動域)
const CAMERA_STEP_DEG: i32 = 2;
const CAMERA_ANGLE_LIMIT: i32 = 45;

// ブザー音
const BEEP_FREQ_HZ: f32 = 2000.0;
const BEEP_DURATION_MS: u64 = 50;
const BEEP_VOLUME: f32 = 0.7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputAction {
    MoveForward,
    MoveBackward,
    SteerLeft,
    SteerRight,
    CameraUp,
    CameraDown,
    CameraLeft,
    CameraRight,
    CameraCenter,
    BeepBuzzer,
}

pub struct RcController {
    action_states: HashMap<InputAction, bool>,
    key_map: HashMap<egui::Key, InputAction>,
    was_beep_pressed: bool,
}

impl RcController {
    pub fn new(key_map: HashMap<egui::Key, InputAction>) -> Self {
        let mut action_states = HashMap::new();
        for action in key_map.values() {
            action_states.insert(*action, false);
        }

        Self {
            action_states,
            key_map,
            was_beep_pressed: false,
        }
    }

    fn is_pressed(&self, action: InputAction) -> bool {
        *self.action_states.get(&action).unwrap_or(&false)
    }

    /// 現在のキー押下状態をeguiから読み取り、action_statesへ反映する。
    /// (Escape/Q終了判定やフォーカス管理はpicarx_sample.rs側の責務)
    pub fn sync_from_egui(&mut self, ctx: &egui::Context) {
        let pressed: Vec<(InputAction, bool)> = ctx.input(|i| {
            self.key_map
                .iter()
                .map(|(key, action)| (*action, i.key_down(*key)))
                .collect()
        });
        for (action, is_down) in pressed {
            self.action_states.insert(action, is_down);
        }
    }

    // 現在の入力状態を計算し、実機（px）に直接命令を書き込むメソッド
    pub fn update_hardware(&mut self, px: &mut Picarx) {
        // 1. モーター制御の計算と命令送信
        let speed = if self.is_pressed(InputAction::MoveForward) {
            MOVE_SPEED
        } else if self.is_pressed(InputAction::MoveBackward) {
            -MOVE_SPEED
        } else {
            0
        };

        if speed > 0 {
            px.forward(speed);
        } else if speed < 0 {
            px.backward(-speed);
        } else {
            px.stop();
        }

        // 2. ステアリング制御の計算と命令送信
        let steer = if self.is_pressed(InputAction::SteerRight) {
            STEER_ANGLE
        } else if self.is_pressed(InputAction::SteerLeft) {
            -STEER_ANGLE
        } else {
            0
        };
        px.set_dir_servo_angle(steer);

        // 3. カメラ角度の計算と命令送信
        // pan/tilt の現在値は Picarx 側を単一の真実の源とし、ここでは保持しない
        let (pan_angle, tilt_angle) = calculate_camera_angles(
            px.pan_angle,
            px.tilt_angle,
            self.is_pressed(InputAction::CameraCenter),
            self.is_pressed(InputAction::CameraUp),
            self.is_pressed(InputAction::CameraDown),
            self.is_pressed(InputAction::CameraLeft),
            self.is_pressed(InputAction::CameraRight),
        );

        px.set_cam_pan_angle(pan_angle);
        px.set_cam_tilt_angle(tilt_angle);

        // Beep the buzzer
        let is_beep_pressed = self.is_pressed(InputAction::BeepBuzzer);
        if is_beep_pressed && !self.was_beep_pressed {
            if let Ok(()) = px.play_tone(BEEP_FREQ_HZ, BEEP_DURATION_MS, BEEP_VOLUME) {
                println!("[Speaker] Played a beep tone.");
            } else {
                println!("[Speaker] Failed to play tone.");
            }
        }
    }
}

fn calculate_camera_angles(
    mut pan_angle: i32,
    mut tilt_angle: i32,
    center: bool,
    up: bool,
    down: bool,
    left: bool,
    right: bool,
) -> (i32, i32) {
    if center {
        return (0, 0);
    }

    if up {
        tilt_angle = (tilt_angle - CAMERA_STEP_DEG).max(-CAMERA_ANGLE_LIMIT);
    }
    if down {
        tilt_angle = (tilt_angle + CAMERA_STEP_DEG).min(CAMERA_ANGLE_LIMIT);
    }
    if right {
        pan_angle = (pan_angle - CAMERA_STEP_DEG).max(-CAMERA_ANGLE_LIMIT);
    }
    if left {
        pan_angle = (pan_angle + CAMERA_STEP_DEG).min(CAMERA_ANGLE_LIMIT);
    }

    (pan_angle, tilt_angle)
}

#[cfg(test)]
mod tests {
    use super::calculate_camera_angles;

    #[test]
    fn camera_center_overrides_directional_input() {
        assert_eq!(
            calculate_camera_angles(20, -20, true, true, true, true, true),
            (0, 0)
        );
    }

    #[test]
    fn camera_angles_are_changed_by_one_step_and_clamped() {
        assert_eq!(
            calculate_camera_angles(0, 0, false, true, false, true, false),
            (2, -2)
        );
        assert_eq!(
            calculate_camera_angles(45, -45, false, true, false, true, false),
            (45, -45)
        );
    }
}
