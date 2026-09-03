use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use super::common::{LlmProvider, ProviderAgent, SYSTEM_PROMPT};

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseCandidate {
    content: Option<GeminiResponseContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePayload {
    candidates: Vec<GeminiResponseCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorBody {
    error: GeminiError,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    code: i32,
    message: String,
    status: String,
}

/// Gemini-specific configuration and HTTP transport.
pub struct GeminiProvider {
    api_key: String,
}

/// Gemini agent with the same non-blocking API as the Ollama agent.
pub type GeminiAgent = ProviderAgent<GeminiProvider>;

/// Gemini's API-specific structured-output schema.
fn command_response_schema() -> Value {
    serde_json::json!({
        "type": "OBJECT",
        "properties": {
            "user_transcription": {
                "type": "STRING",
                "description": "The user's spoken request transcribed into Japanese text. Return exactly what the user said as accurately as possible."
            },
            "commands": {
                "type": "ARRAY",
                "description": "List of robot commands to execute in sequence.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "action": {
                            "type": "STRING",
                            "enum": [
                                "forward", "backward", "steer", "stop",
                                "set_cam_pan", "set_cam_tilt", "get_distance",
                                "get_grayscale", "get_battery_status"
                            ]
                        },
                        "speed": {
                            "type": "INTEGER",
                            "description": "Movement speed from 0 to 100."
                        },
                        "duration_ms": {
                            "type": "INTEGER",
                            "description": "Movement duration in milliseconds. Use this for time-based requests such as 10 seconds, 10s, 10000ms, or 10秒."
                        },
                        "angle": { "type": "INTEGER" },
                        "until": condition_leaf_schema(),
                    },
                    "required": ["action", "speed", "duration_ms"]
                }
            },
            "reply": {
                "type": "STRING",
                "description": "A short, natural reply message in Japanese to the user's request."
            },
            "memory_updates": {
                "type": "OBJECT",
                "description": "Key-value pairs of persistent long-term memory to save (e.g. user.name, robot.name).",
                "nullable": true
            }
        },
        "required": ["user_transcription", "commands", "reply"]
    })
}

fn condition_leaf_schema() -> Value {
    serde_json::json!({
        "type": "OBJECT",
        "properties": {
            "sensor": {
                "type": "STRING",
                "enum": [
                    "distance", "grayscale_left", "grayscale_center",
                    "grayscale_right", "battery_voltage", "battery_percent"
                ]
            },
            "op": {
                "type": "STRING",
                "enum": ["eq", "ne", "gte", "lte", "gt", "lt"]
            },
            "value": { "type": "NUMBER" }
        },
        "required": ["sensor", "op", "value"]
    })
}

fn gemini_prompt(prompt: &str) -> String {
    gemini_prompt_with_context(prompt, "")
}

fn gemini_prompt_with_context(prompt: &str, context: &str) -> String {
    let context_section = if context.trim().is_empty() {
        String::new()
    } else {
        format!("\n\nContext:\n{}\n", context)
    };

    format!(
        "{}\n\nGemini-specific output rules:\n\
         - Return compact JSON only.\n\
         - Do not include null values, empty objects, markdown, or explanations.\n\
         - Convert time requests exactly: 10 seconds / 10秒 => duration_ms: 10000.\n\
         - Do not use sensor fields for time requests.\n\
         - IMPORTANT: Do not output until unless the user explicitly requests a sensor-based stopping condition in their prompt.\n\
         - Do not output an empty until object.\n\
         - When user gives their name, robot's name, or persistent settings, output them in memory_updates.{}\n\n\
         User request: {}",
        SYSTEM_PROMPT, context_section, prompt
    )
}

fn normalize_gemini_response(response_text: &str) -> Result<String, String> {
    let mut value = serde_json::from_str::<Value>(response_text)
        .map_err(|e| format!("Gemini API response was not valid JSON: {}", e))?;

    let Some(commands) = value.get_mut("commands").and_then(Value::as_array_mut) else {
        return Ok(response_text.to_string());
    };

    for command in commands {
        let Some(object) = command.as_object_mut() else {
            continue;
        };

        let empty_until = object
            .get("until")
            .and_then(Value::as_object)
            .is_some_and(|until| until.is_empty());
        if empty_until {
            object.remove("until");
        }

        let until_sensor = object.remove("until_sensor");
        let until_op = object.remove("until_op");
        let until_value = object.remove("until_value");
        let legacy_until_count =
            until_sensor.is_some() as u8 + until_op.is_some() as u8 + until_value.is_some() as u8;
        if legacy_until_count > 0 && legacy_until_count < 3 {
            return Err(
                "Gemini returned incomplete legacy sensor-stop fields; refusing unsafe command"
                    .to_string(),
            );
        }
        if let (Some(sensor), Some(op), Some(value)) = (until_sensor, until_op, until_value) {
            object.insert(
                "until".to_string(),
                serde_json::json!({
                    "sensor": sensor,
                    "op": op,
                    "value": value
                }),
            );
        }
    }

    serde_json::to_string(&value)
        .map_err(|e| format!("Gemini response normalization failed: {}", e))
}

impl Default for GeminiProvider {
    fn default() -> Self {
        Self {
            api_key: load_api_key(),
        }
    }
}

fn load_api_key() -> String {
    // Load the same locations as the original Gemini implementation, while
    // keeping this provider-specific environment handling out of the agent.
    let _ = dotenvy::dotenv();

    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        let _ = dotenvy::from_path(exe_dir.join(".env"));
    }

    let _ = dotenvy::from_path("rust/.env");
    std::env::var("GEMINI_API_KEY").unwrap_or_default()
}

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;

impl GeminiProvider {
    fn send_generate_content(&self, contents_parts: Vec<Value>) -> Result<String, String> {
        if self.api_key.is_empty() || self.api_key == "YOUR_GEMINI_API_KEY_HERE" {
            return Err(
                "GEMINI_API_KEY is not set. Please set your API key in the .env file (same directory as the executable) or set it as a GEMINI_API_KEY environment variable."
                    .to_string(),
            );
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
            self.api_key
        );
        let request_body = serde_json::json!({
            "contents": [{
                "parts": contents_parts
            }],
            "systemInstruction": {
                "parts": [{
                    "text": SYSTEM_PROMPT
                }]
            },
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
                "responseSchema": command_response_schema()
            }
        });

        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(15))
            .build();

        agent
            .post(&url)
            .send_json(&request_body)
            .map_err(|e| match e {
                ureq::Error::Status(_, response) => {
                    if let Ok(error) = response.into_json::<GeminiErrorBody>() {
                        format!(
                            "Gemini API Error {} ({}): {}",
                            error.error.code, error.error.status, error.error.message
                        )
                    } else {
                        "Gemini APIがHTTPエラーを返しました。".to_string()
                    }
                }
                _ => format!("Gemini APIへの通信に失敗しました: {}", e),
            })?
            .into_json::<GeminiResponsePayload>()
            .map_err(|e| format!("Gemini APIのレスポンス解析に失敗: {}", e))
            .and_then(|payload| {
                let candidate = payload.candidates.first().ok_or_else(|| {
                    "Gemini APIから回答（candidates）が得られませんでした。".to_string()
                })?;

                if let Some(reason) = candidate.finish_reason.as_deref()
                    && reason != "STOP"
                {
                    return Err(format!(
                        "Geminiが正常終了しませんでした。finishReason={}",
                        reason
                    ));
                }

                let content = candidate
                    .content
                    .as_ref()
                    .ok_or_else(|| "Gemini APIのレスポンスがありません。".to_string())?;
                let part = content
                    .parts
                    .first()
                    .ok_or_else(|| "Gemini APIのレスポンス（parts）が空です。".to_string())?;

                println!("[DEBUG] Gemini API Response: {}", part.text);
                normalize_gemini_response(&part.text)
            })
    }

    /// 要約などのプレーンテキスト生成リクエスト（スキーマ・システムプロンプトなし）
    pub fn send_text_prompt(&self, prompt_text: &str) -> Result<String, String> {
        if self.api_key.is_empty() || self.api_key == "YOUR_GEMINI_API_KEY_HERE" {
            return Err(
                "GEMINI_API_KEY is not set. Please set your API key in the .env file or set it as a GEMINI_API_KEY environment variable."
                    .to_string(),
            );
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
            self.api_key
        );
        let request_body = serde_json::json!({
            "contents": [{
                "parts": [{ "text": prompt_text }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024
            }
        });

        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(20))
            .build();

        agent
            .post(&url)
            .send_json(&request_body)
            .map_err(|e| format!("Gemini APIへの通信に失敗しました: {}", e))?
            .into_json::<GeminiResponsePayload>()
            .map_err(|e| format!("Gemini APIのレスポンス解析に失敗: {}", e))
            .and_then(|payload| {
                let candidate = payload
                    .candidates
                    .first()
                    .ok_or_else(|| "Gemini APIから回答が得られませんでした。".to_string())?;
                let content = candidate
                    .content
                    .as_ref()
                    .ok_or_else(|| "Gemini APIのレスポンスがありません。".to_string())?;
                let part = content
                    .parts
                    .first()
                    .ok_or_else(|| "Gemini APIのレスポンスが空です。".to_string())?;
                Ok(part.text.clone())
            })
    }
}

impl LlmProvider for GeminiProvider {
    fn request(&self, prompt: &str) -> Result<String, String> {
        let prompt_text = gemini_prompt(prompt);
        self.send_generate_content(vec![serde_json::json!({
            "text": prompt_text
        })])
    }

    fn request_with_context(&self, prompt: &str, context: &str) -> Result<String, String> {
        let prompt_text = gemini_prompt_with_context(prompt, context);
        self.send_generate_content(vec![serde_json::json!({
            "text": prompt_text
        })])
    }

    fn request_audio(&self, wav_data: &[u8]) -> Result<String, String> {
        self.request_audio_with_context(wav_data, "")
    }

    fn request_audio_with_context(&self, wav_data: &[u8], context: &str) -> Result<String, String> {
        let base64_audio = BASE64.encode(wav_data);
        let context_section = if context.trim().is_empty() {
            String::new()
        } else {
            format!("\n\nContext:\n{}\n", context)
        };
        let audio_prompt_guide = format!(
            "{}\n\nGemini-specific output rules:\n\
             - Listen to the attached user voice audio carefully.\n\
             - Interpret the user's spoken instruction and output the corresponding robot control JSON commands.\n\
             - Return compact JSON only.\n\
             - Do not include null values, empty objects, markdown, or explanations.\n\
             - Convert time requests exactly: 10 seconds / 10秒 => duration_ms: 10000.\n\
             - Do not use sensor fields for time requests.\n\
             - IMPORTANT: Do not output until unless the user explicitly requests a sensor-based stopping condition in their prompt.\n\
             - Do not output an empty until object.\n\
             - When user gives their name, robot's name, or persistent settings, output them in memory_updates.{}\n",
            SYSTEM_PROMPT, context_section
        );

        self.send_generate_content(vec![
            serde_json::json!({
                "inlineData": {
                    "mimeType": "audio/wav",
                    "data": base64_audio
                }
            }),
            serde_json::json!({
                "text": audio_prompt_guide
            }),
        ])
    }

    fn request_summarize(&self, prompt: &str) -> Result<String, String> {
        self.send_text_prompt(prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::{command_response_schema, normalize_gemini_response};

    #[test]
    fn schema_omits_json_schema_features_unsupported_by_gemini() {
        let schema = command_response_schema();
        let encoded = serde_json::to_string(&schema).expect("schema should serialize");

        for unsupported in ["$defs", "$ref", "additionalProperties"] {
            assert!(
                !encoded.contains(unsupported),
                "Gemini response schema must not contain {unsupported}"
            );
        }
    }

    #[test]
    fn normalizes_complete_legacy_flat_until_condition() {
        let normalized = normalize_gemini_response(
            r#"{"commands":[{"action":"forward","speed":50,"duration_ms":1000,"until_sensor":"distance","until_op":"lt","until_value":20}]}"#,
        )
        .expect("complete legacy response should normalize");

        assert_eq!(
            normalized,
            r#"{"commands":[{"action":"forward","duration_ms":1000,"speed":50,"until":{"op":"lt","sensor":"distance","value":20}}]}"#
        );
    }

    #[test]
    fn rejects_incomplete_legacy_flat_until_condition() {
        assert!(
            normalize_gemini_response(
                r#"{"commands":[{"action":"forward","speed":50,"until_sensor":"distance","until_value":10000}]}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn removes_empty_until_condition() {
        let normalized = normalize_gemini_response(
            r#"{"commands":[{"action":"forward","speed":50,"duration_ms":1000,"until":{}}]}"#,
        )
        .expect("response should normalize");

        assert_eq!(
            normalized,
            r#"{"commands":[{"action":"forward","duration_ms":1000,"speed":50}]}"#
        );
    }
}
