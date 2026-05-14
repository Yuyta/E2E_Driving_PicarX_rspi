use rppal::gpio::{Gpio, InputPin, OutputPin};
use std::time::{Duration, Instant};
use std::thread;

pub struct Ultrasonic {
    trig: OutputPin,
    echo: InputPin,
    timeout: Duration,
}

impl Ultrasonic {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let gpio = Gpio::new()?;

        // MCURSTピンは BCM 5
        if let Ok(mut mcu_rst) = gpio.get(5).map(|p| p.into_output()) {
            mcu_rst.set_low();
            thread::sleep(Duration::from_millis(10));
            mcu_rst.set_high();
            thread::sleep(Duration::from_millis(10));
        }

        // "D2": 27 (Trig)
        // "D3": 22 (Echo)
        let mut trig = gpio.get(27)?.into_output();
        let echo = gpio.get(22)?.into_input_pulldown();

        trig.set_low();

        Ok(Ultrasonic {
            trig,
            echo,
            timeout: Duration::from_millis(20),
        })
    }

    fn read_once(&mut self) -> f64 {
        // パルス送信
        self.trig.set_low();
        thread::sleep(Duration::from_millis(1)); // 0.001s
        self.trig.set_high();
        thread::sleep(Duration::from_micros(10)); // 0.00001s
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
        let cm = (during * 343.3 / 2.0) * 100.0;

        (cm * 100.0).round() / 100.0
    }

    pub fn read_distance(&mut self) -> f64 {
        // 10回リトライし、成功したらその値を返す
        for _ in 0..10 {
            let dist = self.read_once();
            if dist != -1.0 {
                return dist;
            }
            // 少しだけ休む（次のパルスとの混信を防ぐため）
            thread::sleep(Duration::from_millis(5));
        }
        -1.0
    }
}

