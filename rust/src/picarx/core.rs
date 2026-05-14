use std::thread;
use std::time::Duration;
use std::fs::File;
use std::path::Path;
use std::io::BufReader;
use rppal::i2c::I2c;
use rppal::gpio::{Gpio, OutputPin};
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Player, Decoder, Device, DeviceTrait};
use rodio::source::{SineWave, Source};
use rodio::cpal::traits::HostTrait;
use crate::picarx::ultrasonic::Ultrasonic;

const HAT_ADDR: u16 = 0x14;

// GrayScale
const REG_ADC_BASE: u8 = 0x10;  // A0=0x10, A1=0x11, A2=0x12...

// PWM channel register (for servo)
const REG_PWM_CH0: u8 = 0x20;  // サーボ P0 (Pan)
const REG_PWM_CH1: u8 = 0x21;  // サーボ P1 (Tilt)
const REG_PWM_CH2: u8 = 0x22;  // サーボ P2 (Steering)

// [robothat v4: mode1] ドライブモーターの速度制御PWMレジスタ
const REG_MOTOR1_PWM: u8 = 0x20 + 13;  // P13 (左モーター速度)
const REG_MOTOR2_PWM: u8 = 0x20 + 12;  // P12 (右モーター速度)

// Speaker ON/OFF register
const PIN_SPEAKER_EN: u8 = 20;

// タイマー設定レジスタ
const REG_PRESCALER_T0: u8 = 0x40;
const REG_PERIOD_T0: u8 = 0x44;

pub struct Picarx {
    i2c: I2c,
    _servo_en: OutputPin,
    _mcu_rst: OutputPin,
    _speaker_en: OutputPin,
    pub dir_angle: i32,
    pub pan_angle: i32,
    pub tilt_angle: i32,

    // ラズパイの物理GPIOから直接伸びている方向制御ピン (D4, D5)
    motor1_dir_pin: OutputPin, // 左モーター方向 (D4 -> BCM 23)
    motor2_dir_pin: OutputPin, // 右モーター方向 (D5 -> BCM 24)

    // モーター反転設定（Python仕様に準拠）
    left_reverse: bool,
    right_reverse: bool,

    ultrasonic: Ultrasonic,
    _audio_sink: MixerDeviceSink,
    audio_player: Player,
}

impl Picarx {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let gpio = Gpio::new()?;

        // MCU物理リセット
        let mut mcu_rst = gpio.get(5)?.into_output();
        mcu_rst.set_low();
        thread::sleep(Duration::from_millis(20));
        mcu_rst.set_high();
        thread::sleep(Duration::from_millis(50));

        // サーボ物理電源ON (High)
        let mut servo_en = gpio.get(26)?.into_output();
        servo_en.set_high();

        // モーターの方向ピン (D4=GPIO23, D5=GPIO24) を初期化
        let motor1_dir_pin = gpio.get(23)?.into_output();
        let motor2_dir_pin = gpio.get(24)?.into_output();

        // アンプスピーカーを初期化 (OFF)
        let mut speaker_en = gpio.get(PIN_SPEAKER_EN)?.into_output();
        speaker_en.set_low();

        let audio_device = Self::find_audio_device()?;

        let builder = DeviceSinkBuilder::from_device(audio_device)?;
        let mut audio_sink = builder.open_stream()?;
        audio_sink.log_on_drop(false);
        let audio_player = Player::connect_new(audio_sink.mixer());

        let mut i2c = I2c::new()?;
        i2c.set_slave_address(HAT_ADDR)?;

        let mut s = Self {
            i2c,
            _servo_en: servo_en,
            _mcu_rst: mcu_rst,
            _speaker_en: speaker_en,
            dir_angle: 0,
            pan_angle: 0,
            tilt_angle: 0,
            motor1_dir_pin,
            motor2_dir_pin,
            // Pythonのデフォルト設定通り、右側だけを反転に
            left_reverse: false,
            right_reverse: true,
            ultrasonic: Ultrasonic::new()?,
            _audio_sink: audio_sink,
            audio_player,
        };

        // スピーカーをONにし、無音を鳴らす (ノイズクリア)
        s._speaker_en.set_high();
        let _ = s.play_tone(440.0, 10, 0.0);

        // 4. タイマー初期化（50Hz, PERIOD=4095, PRESCALER=351）
        s.init_hat_timers()?;

        // 5. 起動時初期位置リセット
        s.stop();
        s.set_dir_servo_angle(0);
        s.set_cam_pan_angle(0);
        s.set_cam_tilt_angle(0);

        Ok(s)
    }

    fn find_audio_device() -> Result<Device, Box<dyn std::error::Error>> {
        let host = rodio::cpal::default_host();

        let device = host.output_devices()?
            .find(|d| {
                d.description()
                    .map(|desc| {
                        desc.name() == "robothat"
                    })
                    .unwrap_or(false)
            })
            .or_else(|| host.default_output_device());

        let device = device.ok_or("利用可能なオーディオデバイスが見つかりません")?;
        Ok(device)
    }

    fn init_hat_timers(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let period: u16 = 4095;
        let prescaler: u16 = 351;

        for i in 0..4 {
            self.write_word(REG_PRESCALER_T0 + i, prescaler)?;
            thread::sleep(Duration::from_millis(1));
            self.write_word(REG_PERIOD_T0 + i, period)?;
            thread::sleep(Duration::from_millis(1));
        }

        thread::sleep(Duration::from_millis(50));
        println!("✓ Timers initialized (50Hz: ARR={}, PSC={})", period, prescaler);
        Ok(())
    }

    fn write_word(&mut self, reg: u8, value: u16) -> Result<(), Box<dyn std::error::Error>> {
        let high_byte = ((value >> 8) & 0xFF) as u8;
        let low_byte = (value & 0xFF) as u8;
        self.i2c.write(&[reg, high_byte, low_byte])?;  // big endian
        Ok(())
    }

    fn read_adc_channel(&mut self, channel: u8) -> Result<u16, Box<dyn std::error::Error>> {
        // robot-hat 仕様準拠
        // channel 番号を反転させてレジスタを作る (chn = 7 - chn | 0x10)
        let inverted_chn = (7 - channel) | REG_ADC_BASE;

        // [channel, 0, 0] の3バイトをコマンドとして一度完全に書き込む
        self.i2c.write(&[inverted_chn, 0, 0])?;

        let mut buf = [0u8; 2];
        self.i2c.read(&mut buf)?;

        let value = ((buf[0] as u16) << 8) | (buf[1] as u16);
        Ok(value)
    }

    fn set_motor_speed(&mut self, motor_idx: u8, speed: i32) {
        let mut clamped_speed = speed.clamp(-100, 100);

        let is_left = motor_idx == 0;
        let is_reverse = if is_left { self.left_reverse } else { self.right_reverse };
        if is_reverse {
            clamped_speed = -clamped_speed;
        }

        let dir_signal = clamped_speed < 0;

        // ラズパイの物理GPIOへ「方向信号」を出力
        if is_left {
            if dir_signal { self.motor1_dir_pin.set_high(); } else { self.motor1_dir_pin.set_low(); }
        } else {
            if dir_signal { self.motor2_dir_pin.set_high(); } else { self.motor2_dir_pin.set_low(); }
        }

        // 速度を絶対値にしてI2CでPWM出力 (0 〜 4095)
        let abs_speed = clamped_speed.abs();
        let pwm_value = ((abs_speed as f32 / 100.0) * 4095.0) as u16;
        let reg_pwm = if is_left { REG_MOTOR1_PWM } else { REG_MOTOR2_PWM };

        let _ = self.write_word(reg_pwm, pwm_value);
    }

    fn set_servo_angle(&mut self, reg: u8, angle: i32) {
        let clamped_angle = angle.clamp(-90, 90);
        // 角度（-90〜90）をパルス幅（500〜2500μs）にマッピング
        let ratio = (clamped_angle + 90) as f32 / 180.0;
        let pulse_us = 500.0 + (ratio * 2000.0);
        // 周期20000μsに対する4095カウントの割合でDuty比を決定
        let pwm_value = (pulse_us * 4095.0) / 20000.0;
        let _ = self.write_word(reg, pwm_value as u16);
    }

    pub fn forward(&mut self, speed: i32) {
        self.set_motor_speed(0, speed);
        self.set_motor_speed(1, speed);
    }

    pub fn backward(&mut self, speed: i32) {
        self.forward(-speed);
    }

    pub fn stop(&mut self) {
        self.set_motor_speed(0, 0);
        self.set_motor_speed(1, 0);
    }

    pub fn set_dir_servo_angle(&mut self, angle: i32) {
        self.dir_angle = angle;
        self.set_servo_angle(REG_PWM_CH2, angle);
    }

    pub fn set_cam_pan_angle(&mut self, angle: i32) {
        self.pan_angle = angle;
        self.set_servo_angle(REG_PWM_CH0, angle);
    }

    pub fn set_cam_tilt_angle(&mut self, angle: i32) {
        self.tilt_angle = angle;
        self.set_servo_angle(REG_PWM_CH1, angle);
    }

    pub fn get_grayscale_data(&mut self) -> Result<[u16; 3], Box<dyn std::error::Error>> {
        let a0 = self.read_adc_channel(0)?;
        let a1 = self.read_adc_channel(1)?;
        let a2 = self.read_adc_channel(2)?;

        Ok([a0 & 0xFFF, a1 & 0xFFF, a2 & 0xFFF])
    }

    pub fn get_distance(&mut self) -> f64 {
        self.ultrasonic.read_distance()
    }

    /// バッテリーの実際の電圧(V)を取得する
    pub fn get_battery_voltage(&mut self) -> Result<f64, Box<dyn std::error::Error>> {
        // 1. バッテリー監視用のA4（チャンネル4）から生のADC値(0〜4095)を取得
        let raw_val = self.read_adc_channel(4)? & 0xFFF;

        // 2. 生の値を、ADCピンの電圧（最大3.3V）に変換
        let adc_voltage = (raw_val as f64 / 4095.0) * 3.3;

        // 3. 回路上で1/3に分圧されているため、3倍して元のバッテリー電圧に戻す
        let battery_voltage = adc_voltage * 3.0;

        // 4. 小数第2位で丸めて返す
        Ok((battery_voltage * 100.0).round() / 100.0)
    }

    /// バッテリーの電圧(V)と残量目安(%)を同時に取得する
    /// 返り値: Result<(電圧, パーセンテージ), Error>
    pub fn get_battery_status(&mut self) -> Result<(f64, i32), Box<dyn std::error::Error>> {
        let voltage = self.get_battery_voltage()?;

        // 7.4Vリチウムイオンバッテリー（2セル）の一般的な電圧カーブ
        // 満充電: 8.4V, 放電限界: 6.8V
        let percentage = if voltage >= 8.4 {
            100
        } else if voltage <= 6.8 {
            0
        } else {
            // 6.8V〜8.4V（幅 1.6V）の間で線形マッピング
            ((voltage - 6.8) / (8.4 - 6.8) * 100.0) as i32
        };

        // (電圧, %) のタプルで返す
        Ok((voltage, percentage))
    }

    /// 指定した周波数(Hz), 時間(ms), ボリューム[0.0, 1.0] で「ピーッ」という電子音を生成して鳴らす
    pub fn play_tone(&mut self, frequency: f32, duration_ms: u64, volume: f32) -> Result<(), Box<dyn std::error::Error>> {
        let source = SineWave::new(frequency)
            .take_duration(Duration::from_millis(duration_ms))
            .amplify(volume);

        // 保持しているプレイヤーに音源を追加して待機
        self.audio_player.append(source);
        self.audio_player.sleep_until_end();

        Ok(())
    }

    /// ラズパイ内にある音声ファイル (wav, mp3, ogg 等) を鳴らす
    pub fn play_file<P: AsRef<Path>>(&mut self, file_path: P, volume: f32) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let source = Decoder::new(reader)?
            .amplify(volume);

        // 保持しているプレイヤーに音源を追加して待機
        self.audio_player.append(source);
        self.audio_player.sleep_until_end();

        Ok(())
    }
}

impl Drop for Picarx {
    fn drop(&mut self) {
        self.stop();
        self.set_dir_servo_angle(0);
        self.set_cam_pan_angle(0);
        self.set_cam_tilt_angle(0);

        // speaker off
        self._speaker_en.set_low();

        // servo off
        std::thread::sleep(std::time::Duration::from_millis(100));
        self._servo_en.set_low();
    }
}

