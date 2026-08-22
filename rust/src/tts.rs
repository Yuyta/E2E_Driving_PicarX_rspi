use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsEngine {
    Default,
    PiperPlus,
}

/// テキストの言語を自動判別します。
/// 日本語のひらがな・カタカナ・漢字が1文字でも含まれていれば "ja"、それ以外は "en" を返します。
pub fn detect_language(text: &str) -> &'static str {
    let has_japanese = text.chars().any(|c| {
        ('\u{3040}'..='\u{309F}').contains(&c) || // ひらがな
        ('\u{30A0}'..='\u{30FF}').contains(&c) || // カタカナ
        ('\u{4E00}'..='\u{9FAF}').contains(&c) // 漢字
    });

    if has_japanese {
        "ja"
    } else {
        "en"
    }
}

fn preview_file_start(path: &Path) -> String {
    match fs::read(path) {
        Ok(bytes) => {
            let preview: Vec<u8> = bytes.into_iter().take(80).collect();
            String::from_utf8_lossy(&preview)
                .chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect::<String>()
        }
        Err(error) => format!("failed to read preview: {}", error),
    }
}

/// 指定されたテキストと言語、エンジン設定に基づいて音声ファイルを生成します。
/// 生成されたファイルは `/tmp/picarx_tts_<timestamp>.wav` に保存されます。
pub fn generate_tts_file(
    text: &str,
    lang: &str,
    engine: TtsEngine,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let output_path = PathBuf::from(format!("/tmp/picarx_tts_{}.wav", timestamp));

    match engine {
        TtsEngine::Default => {
            if lang == "ja" {
                let voice_path =
                    if Path::new("/usr/share/hts-voice/mei/mei_normal.htsvoice").exists() {
                        "/usr/share/hts-voice/mei/mei_normal.htsvoice"
                    } else {
                        "/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice"
                    };

                // open_jtalk コマンド実行 (標準入力からテキストを受け取る)
                let mut child = Command::new("open_jtalk")
                    .arg("-x")
                    .arg("/var/lib/mecab/dic/open-jtalk/naist-jdic")
                    .arg("-m")
                    .arg(voice_path)
                    .arg("-ow")
                    .arg(&output_path)
                    .stdin(Stdio::piped())
                    .spawn()?;

                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(text.as_bytes())?;
                }
                let status = child.wait()?;
                if !status.success() {
                    return Err(format!("open_jtalk failed with status: {:?}", status).into());
                }
            } else {
                // espeak-ng コマンド実行 (標準入力からテキストを受け取る)
                let mut child = Command::new("espeak-ng")
                    .arg("-w")
                    .arg(&output_path)
                    .stdin(Stdio::piped())
                    .spawn()?;

                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(text.as_bytes())?;
                }
                let status = child.wait()?;
                if !status.success() {
                    return Err(format!("espeak-ng failed with status: {:?}", status).into());
                }
            }
        }
        TtsEngine::PiperPlus => {
            let install_dir = Path::new("/usr/local/share/piper-plus");
            let cmd_candidates = [
                "/usr/local/share/piper-plus/bin/piper-plus",
                "/usr/local/share/piper-plus/bin/piper",
                "/usr/local/share/piper-plus/piper-plus/bin/piper-plus",
                "/usr/local/share/piper-plus/piper/bin/piper",
            ];
            let cmd = cmd_candidates
                .iter()
                .copied()
                .find(|path| Path::new(path).exists())
                .ok_or_else(|| {
                    format!(
                        "piper executable not found. checked: {}",
                        cmd_candidates.join(", ")
                    )
                })?;

            let model_path = install_dir.join("models/tsukuyomi.onnx");
            let config_path = install_dir.join("models/config.json");
            if !model_path.exists() {
                return Err(format!("piper model not found: {}", model_path.display()).into());
            }
            if !config_path.exists() {
                return Err(format!("piper config not found: {}", config_path.display()).into());
            }
            let config_text = fs::read_to_string(&config_path)?;
            if let Err(error) = serde_json::from_str::<serde_json::Value>(&config_text) {
                return Err(format!(
                    "piper config is not valid JSON: {}. path: {}. first bytes: {:?}",
                    error,
                    config_path.display(),
                    preview_file_start(&config_path)
                )
                .into());
            }

            let mut child = Command::new(cmd)
                .current_dir(install_dir)
                .env("LANG", "C.UTF-8")
                .env("LC_ALL", "C.UTF-8")
                .arg("--model")
                .arg(&model_path)
                .arg("--config")
                .arg(&config_path)
                .arg("--text")
                .arg(text)
                .arg("--language")
                .arg("ja-en-zh-es-fr-pt")
                .arg("--speaker")
                .arg("0")
                .arg("--noise-scale")
                .arg("0.667")
                .arg("--length-scale")
                .arg("1.5")
                .arg("--output_file")
                .arg(&output_path)
                .spawn()?;

            let status = child.wait()?;
            if !status.success() {
                return Err(format!("piper failed with status: {:?}", status).into());
            }
        }
    }

    Ok(output_path)
}
