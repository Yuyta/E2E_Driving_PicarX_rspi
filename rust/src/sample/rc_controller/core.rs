use std::collections::HashMap;
use show_image::event::{WindowEvent, VirtualKeyCode, ElementState};
use crate::picarx::Picarx;

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
    GetGrayscale,
    GetDistance,
    GetBatteryStatus,
    BeepBuzzer,
}

pub struct RcController {
    action_states: HashMap<InputAction, bool>,
    key_map: HashMap<VirtualKeyCode, InputAction>,
    pan_angle: i32,
    tilt_angle: i32,
    was_grayscale_pressed: bool,
    was_distance_pressed: bool,
    was_battery_pressed: bool,
    was_beep_pressed: bool,
}

impl RcController {
    pub fn new(key_map: HashMap<VirtualKeyCode, InputAction>) -> Self {
        let mut action_states = HashMap::new();
        for action in key_map.values() {
            action_states.insert(*action, false);
        }

        Self {
            action_states,
            key_map,
            pan_angle: 0,
            tilt_angle: 0,
            was_grayscale_pressed: false,
            was_distance_pressed: false,
            was_battery_pressed: false,
            was_beep_pressed: false,
        }
    }

    fn is_pressed(&self, action: InputAction) -> bool {
        *self.action_states.get(&action).unwrap_or(&false)
    }

    pub fn process_event(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::KeyboardInput(key_event) = event {
            let is_pressed = key_event.input.state == ElementState::Pressed;

            if let Some(key) = key_event.input.key_code {
                if is_pressed && (key == VirtualKeyCode::Escape || key == VirtualKeyCode::Q) {
                    return false;
                }

                if let Some(action) = self.key_map.get(&key) {
                    self.action_states.insert(*action, is_pressed);
                }
            }
        }
        true
    }

    // 現在の入力状態を計算し、実機（px）に直接命令を書き込むメソッド
    pub fn update_hardware(&mut self, px: &mut Picarx) {
        // 1. モーター制御の計算と命令送信
        let speed = if self.is_pressed(InputAction::MoveForward) {
            80
        } else if self.is_pressed(InputAction::MoveBackward) {
            -80
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
            35
        } else if self.is_pressed(InputAction::SteerLeft) {
            -35
        } else {
            0
        };
        px.set_dir_servo_angle(steer);

        // 3. カメラ角度の計算と命令送信
        if self.is_pressed(InputAction::CameraCenter) {
            self.pan_angle  = 0;
            self.tilt_angle = 0;
        } else {
            if self.is_pressed(InputAction::CameraUp)    { self.tilt_angle = (self.tilt_angle - 2).min(-45); }
            if self.is_pressed(InputAction::CameraDown)  { self.tilt_angle = (self.tilt_angle + 2).max( 45); }
            if self.is_pressed(InputAction::CameraRight) { self.pan_angle  = (self.pan_angle  - 2).min(-45); }
            if self.is_pressed(InputAction::CameraLeft)  { self.pan_angle  = (self.pan_angle  + 2).max( 45); }
        }

        px.set_cam_pan_angle(self.pan_angle);
        px.set_cam_tilt_angle(self.tilt_angle);

        // GrayScale Sensor
        let is_grayscale_pressed = self.is_pressed(InputAction::GetGrayscale);
        if is_grayscale_pressed && !self.was_grayscale_pressed {
            if let Ok(values) = px.get_grayscale_data() {
                println!("[Grayscale Sensor] A0(left): {:4} | A1(center): {:4} | A2(right): {:4}", values[0], values[1], values[2]);
            } else {
                println!("[Grayscale Sensor] failed to get grayscale-data.");
            }
        }
        self.was_grayscale_pressed = is_grayscale_pressed;

        // Ultrasonic Sensor
        let is_distance_pressed = self.is_pressed(InputAction::GetDistance);
        if is_distance_pressed && !self.was_distance_pressed {
            let distance = px.get_distance();
            if distance < 0.0 {
                println!("[Ultrasonic Sensor] Timeout (Object too far or hardware error)");
            } else {
                println!("[Ultrasonic Sensor] Distance: {:.2} cm", distance);
            }
        }
        self.was_distance_pressed = is_distance_pressed;

        // BatterStatus
        let is_battery_pressed = self.is_pressed(InputAction::GetBatteryStatus);
        if is_battery_pressed && !self.was_battery_pressed {
            if let Ok((voltage, percent)) = px.get_battery_status() {
                if voltage < 7.0 {
                    println!("[BatteryStatus] Voltage: {:.2} V ({}%) : Battery Low!", voltage, percent);
                } else {
                    println!("[BatteryStatus] Voltage: {:.2} V ({}%)", voltage, percent);
                }
            } else {
                println!("[BatteryStatus] failed to get battery-status.");
            }
        }

        // Beep the buzzer
        let is_beep_pressed = self.is_pressed(InputAction::BeepBuzzer);
        if is_beep_pressed && !self.was_beep_pressed {
            // 2000Hz, 50ms
            if let Ok(()) = px.play_tone(2000.0, 50, 0.7) {
                println!("[Speaker] Played a beep tone.");
            } else {
                println!("[Speaker] Failed to play tone.");
            }
        }
    }
}

