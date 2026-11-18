//! 舵角と前方の距離から走行の速度を決める（PC の `safety/speed_limiter.py` と同じ式）。
//!
//! 直進は巡航の速度、最大の舵角と停止の手前の距離では、確実に動く最低の速度まで落とす。
//! 遅い方を使う。

/// 確実に動く最低の速度。30 では止まり、35 では這う（2026-11 の実機計測）
pub const MIN_MOVING_SPEED: i32 = 40;
/// 巡航の速度
pub const DRIVE_SPEED: i32 = 50;

#[derive(Debug, Clone, Copy)]
pub struct SpeedLimiter {
    pub base_speed: i32,
    pub min_speed: i32,
    pub curve_factor: f64,
    pub safe_distance_cm: f64,
    pub slowdown_distance_cm: f64,
}

impl SpeedLimiter {
    pub fn new(base_speed: i32, min_speed: i32) -> Result<Self, String> {
        if min_speed > base_speed {
            return Err(format!(
                "min_speed ({min_speed}) は base_speed ({base_speed}) 以下にしてください"
            ));
        }
        Ok(Self {
            base_speed: base_speed.clamp(0, 100),
            min_speed: min_speed.clamp(0, 100),
            curve_factor: 1.0,
            safe_distance_cm: 15.0,
            slowdown_distance_cm: 40.0,
        })
    }

    /// 舵角（-1〜1）による速度。直進で base_speed、最大の舵角で min_speed
    pub fn curve_speed(&self, steering: f64) -> i32 {
        let penalty = steering.abs().min(1.0).powf(self.curve_factor).min(1.0);
        let drop = f64::from(self.base_speed - self.min_speed) * penalty;
        (f64::from(self.base_speed) - drop).round() as i32
    }

    /// 前方の距離による速度。遠いか分からない時は base_speed
    pub fn obstacle_speed(&self, distance_cm: Option<f64>) -> i32 {
        let Some(d) = distance_cm else {
            return self.base_speed;
        };
        if d >= self.slowdown_distance_cm {
            return self.base_speed;
        }
        if d <= self.safe_distance_cm {
            return self.min_speed;
        }
        let ratio =
            (d - self.safe_distance_cm) / (self.slowdown_distance_cm - self.safe_distance_cm);
        (f64::from(self.min_speed) + f64::from(self.base_speed - self.min_speed) * ratio).round()
            as i32
    }

    pub fn calculate(&self, steering: f64, distance_cm: Option<f64>) -> i32 {
        self.curve_speed(steering)
            .min(self.obstacle_speed(distance_cm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_and_far_drive_at_the_base_speed() {
        let s = SpeedLimiter::new(50, 40).unwrap();
        assert_eq!(s.calculate(0.0, None), 50);
        assert_eq!(s.calculate(0.0, Some(100.0)), 50);
    }

    #[test]
    fn full_lock_or_close_obstacles_slow_to_the_minimum() {
        let s = SpeedLimiter::new(50, 40).unwrap();
        assert_eq!(s.calculate(1.0, None), 40);
        assert_eq!(s.calculate(-1.5, None), 40);
        assert_eq!(s.calculate(0.0, Some(10.0)), 40);
        assert_eq!(s.calculate(0.5, Some(27.5)), 45);
    }

    #[test]
    fn min_speed_above_base_is_rejected() {
        assert!(SpeedLimiter::new(40, 50).is_err());
    }
}
