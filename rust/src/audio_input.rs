use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{SystemTime, UNIX_EPOCH};

/// 音声認識(STT)インターフェース
/// 将来的にVoskなどのローカルSTTエンジンを追加する際にも共通で使用可能
pub trait SpeechToText: Send + Sync {
    fn transcribe(&self, wav_data: &[u8]) -> Result<String, String>;
}

/// プッシュ・トゥ・トーク (PTT) 録音管理
pub struct AudioRecorder {
    child_process: Option<Child>,
    temp_wav_path: Option<PathBuf>,
    sample_rate: u32,
    channels: u16,
}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            child_process: None,
            temp_wav_path: None,
            sample_rate: 16000,
            channels: 1,
        }
    }

    /// 録音中かどうかを返します
    pub fn is_recording(&self) -> bool {
        self.child_process.is_some()
    }

    /// 録音を開始します（非同期サブプロセス `arecord` を起動）
    pub fn start_recording(&mut self) -> Result<(), String> {
        if self.is_recording() {
            return Err("既に録音中です".to_string());
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let wav_path = PathBuf::from(format!("/tmp/picarx_record_{}.wav", timestamp));

        // arecord -q -f S16_LE -r 16000 -c 1 -t wav /tmp/...
        #[cfg(target_os = "linux")]
        {
            let child = Command::new("arecord")
                .arg("-q")
                .arg("-f")
                .arg("S16_LE")
                .arg("-r")
                .arg(self.sample_rate.to_string())
                .arg("-c")
                .arg(self.channels.to_string())
                .arg("-t")
                .arg("wav")
                .arg(&wav_path)
                .spawn()
                .map_err(|e| format!("arecordの起動に失敗しました: {}", e))?;

            self.child_process = Some(child);
            self.temp_wav_path = Some(wav_path);
            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            // 非Linux(Windows等での開発時)のフォールバック
            self.temp_wav_path = Some(wav_path);
            Err("非Linux環境ではarecordによる録音は利用できません".to_string())
        }
    }

    /// 録音を停止し、生成されたWAVファイルのバイナリデータを返します
    pub fn stop_recording(&mut self) -> Result<Vec<u8>, String> {
        let mut child = self
            .child_process
            .take()
            .ok_or_else(|| "録音プロセスが起動していません".to_string())?;

        let wav_path = self
            .temp_wav_path
            .take()
            .ok_or_else(|| "一時録音ファイルパスが存在しません".to_string())?;

        // プロセスにSIGTERM/killを送信して終了させる
        let _ = child.kill();
        let _ = child.wait();

        // WAVファイルを読み込み
        if !wav_path.exists() {
            return Err(format!(
                "録音WAVファイルが見つかりません: {}",
                wav_path.display()
            ));
        }

        let bytes =
            fs::read(&wav_path).map_err(|e| format!("録音WAVファイルの読み込みに失敗: {}", e))?;

        // 一時ファイルの削除
        let _ = fs::remove_file(&wav_path);

        if bytes.is_empty() {
            return Err("録音データが空です".to_string());
        }

        Ok(bytes)
    }

    /// 録音をキャンセルし、一時ファイルを破棄します
    pub fn cancel_recording(&mut self) {
        if let Some(mut child) = self.child_process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(wav_path) = self.temp_wav_path.take() {
            let _ = fs::remove_file(wav_path);
        }
    }
}

impl Default for AudioRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AudioRecorder {
    fn drop(&mut self) {
        self.cancel_recording();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_initial_state() {
        let recorder = AudioRecorder::new();
        assert!(!recorder.is_recording());
    }
}
