use rppal::gpio::{Gpio, InputPin, OutputPin};
use std::time::{Duration, Instant};
use std::thread;

// GPIOピン番号 (BCM)
const PIN_MCU_RST: u8 = 5;
const PIN_TRIG: u8 = 27; // "D2"
const PIN_ECHO: u8 = 22; // "D3"

// MCUリセット時のウェイト
const MCU_RESET_LOW_MS: u64 = 10;
const MCU_RESET_HIGH_MS: u64 = 10;

// エコー待ちタイムアウト
const ECHO_TIMEOUT_MS: u64 = 20;

// トリガーパルス
const TRIG_SETTLE_MS: u64 = 1;
const TRIG_PULSE_US: u64 = 10;

// 音速 (m/s, 常温)
const SOUND_SPEED_M_S: f64 = 343.3;

// 測定リトライ設定
const RETRY_COUNT: u32 = 10;
const RETRY_DELAY_MS: u64 = 5;

pub struct Ultrasonic {
    trig: OutputPin,
    echo: InputPin,
    timeout: Duration,
}

impl Ultrasonic {
    /// `Picarx`経由で使う場合など、呼び出し元が既にMCUリセット済みのときは
    /// `reset_mcu = false`を渡してここでの再リセットをスキップする。
    pub fn new(reset_mcu: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let gpio = Gpio::new()?;

        if reset_mcu {
            match gpio.get(PIN_MCU_RST).map(|p| p.into_output()) {
                Ok(mut mcu_rst) => {
                    mcu_rst.set_low();
                    thread::sleep(Duration::from_millis(MCU_RESET_LOW_MS));
                    mcu_rst.set_high();
                    thread::sleep(Duration::from_millis(MCU_RESET_HIGH_MS));
                }
                Err(e) => {
                    eprintln!("⚠ MCU_RSTピン(BCM {})の取得に失敗（他で使用中の可能性）: {}", PIN_MCU_RST, e);
                }
            }
        }

        let mut trig = gpio.get(PIN_TRIG)?.into_output();
        let echo = gpio.get(PIN_ECHO)?.into_input_pulldown();

        trig.set_low();

        Ok(Ultrasonic {
            trig,
            echo,
            timeout: Duration::from_millis(ECHO_TIMEOUT_MS),
        })
    }

    fn read_once(&mut self) -> f64 {
        // パルス送信
        self.trig.set_low();
        thread::sleep(Duration::from_millis(TRIG_SETTLE_MS));
        self.trig.set_high();
        thread::sleep(Duration::from_micros(TRIG_PULSE_US));
        self.trig.set_low();

        let timeout_start = Instant::now();
        let mut pulse_start = Instant::now();
        let mut pulse_end = Instant::now();

        // EchoがHIGHになるのを待つ (while self.echo.gpio.value == 0:)
        while self.echo.is_low() {
            pulse_start = Instant::now();
            if timeout_start.elapsed() > self.timeout {
                return -1.0;
            }
        }

        // EchoがLOWに戻るのを待つ (while self.echo.gpio.value == 1:)
        while self.echo.is_high() {
            pulse_end = Instant::now();
            if timeout_start.elapsed() > self.timeout {
                return -1.0;
            }
        }

        let during = pulse_end.duration_since(pulse_start).as_secs_f64();
        let cm = (during * SOUND_SPEED_M_S / 2.0) * 100.0;

        (cm * 100.0).round() / 100.0
    }

    pub fn read_distance(&mut self) -> f64 {
        // リトライし、成功したらその値を返す
        for _ in 0..RETRY_COUNT {
            let dist = self.read_once();
            if dist != -1.0 {
                return dist;
            }
            // 少しだけ休む（次のパルスとの混信を防ぐため）
            thread::sleep(Duration::from_millis(RETRY_DELAY_MS));
        }
        -1.0
    }

    /// リトライなしで1回だけ計測する(最大でもECHO_TIMEOUT_MS程度しかブロックしない)。
    /// バックグラウンドでの継続的なポーリング用途で、Picarxを共有するMutexの
    /// ロック保持時間を短く抑えるために使う。
    pub fn try_read_distance_once(&mut self) -> Option<f64> {
        let dist = self.read_once();
        if dist < 0.0 { None } else { Some(dist) }
    }
}

