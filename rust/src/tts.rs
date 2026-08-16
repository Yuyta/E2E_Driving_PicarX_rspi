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
        ('\u{4E00}'..='\u{9FAF}').contains(&c)    // 漢字
    });

    if has_japanese {
        "ja"
    } else {
        "en"
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
                let voice_path = if Path::new("/usr/share/hts-voice/mei/mei_normal.htsvoice").exists() {
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
            // piper-plus の絶対パスを優先し、存在しなければ PATH から探す
            let cmd_path = "/usr/local/share/piper-plus/bin/piper-plus";
            let cmd = if Path::new(cmd_path).exists() {
                cmd_path
            } else {
                "piper-plus"
            };

            let mut child = Command::new(cmd)
                .arg("--model")
                .arg("tsukuyomi")
                .arg("--noise-scale")
                .arg("0.5")
                .arg("--output_file")
                .arg(&output_path)
                .stdin(Stdio::piped())
                .spawn()?;

            if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(text.as_bytes())?;
            }
            let status = child.wait()?;
            if !status.success() {
                return Err(format!("piper-plus failed with status: {:?}", status).into());
            }
        }
    }

    Ok(output_path)
}
