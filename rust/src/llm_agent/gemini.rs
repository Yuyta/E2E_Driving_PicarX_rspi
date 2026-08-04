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
                        "speed": { "type": "INTEGER" },
                        "duration_ms": { "type": "INTEGER" },
                        "angle": { "type": "INTEGER" },
                        "until": condition_leaf_schema(),
                        "until_sensor": {
                            "type": "STRING",
                            "description": "Sensor used for a simple stop condition. Omit unless the user asks for sensor-based stopping.",
                            "enum": [
                                "distance", "grayscale_left", "grayscale_center",
                                "grayscale_right", "battery_voltage", "battery_percent"
                            ]
                        },
                        "until_op": {
                            "type": "STRING",
                            "description": "Comparison operator for until_sensor.",
                            "enum": ["eq", "ne", "gte", "lte", "gt", "lt"]
                        },
                        "until_value": {
                            "type": "NUMBER",
                            "description": "Numeric threshold for until_sensor."
                        },
                        "until_logic": {
                            "type": "STRING",
                            "description": "Logical operator for combining until_conditions.",
                            "enum": ["and", "or", "xor", "not"]
                        },
                        "until_conditions": {
                            "type": "ARRAY",
                            "description": "Sensor conditions to combine with until_logic.",
                            "items": condition_leaf_schema()
                        },
                    },
                    "required": ["action"]
                }
            }
        },
        "required": ["commands"]
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
    format!(
        "{}\n\nGemini-specific output rules:\n\
         - Return compact JSON only.\n\
         - Do not include null values, empty objects, markdown, or explanations.\n\
         - For simple driving, omit sensor stop fields.\n\
         - For a single sensor stop condition, prefer until_sensor, until_op, and until_value.\n\
         - For combined sensor stop conditions, use until_logic and until_conditions.\n\
         - Do not output an empty until object.\n\n\
         User request: {}",
        SYSTEM_PROMPT, prompt
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

        let until_logic = object.remove("until_logic");
        let until_conditions = object.remove("until_conditions");
        if let (Some(Value::String(logic)), Some(conditions)) = (until_logic, until_conditions) {
            match logic.as_str() {
                "and" | "or" | "xor" => {
                    object.insert(
                        "until".to_string(),
                        serde_json::json!({ logic: conditions }),
                    );
                }
                "not" => {
                    let first_condition = conditions
                        .as_array()
                        .and_then(|conditions| conditions.first())
                        .cloned();
                    if let Some(condition) = first_condition {
                        object.insert("until".to_string(), serde_json::json!({ "not": condition }));
                    }
                }
                _ => {}
            }
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

impl LlmProvider for GeminiProvider {
    fn request(&self, prompt: &str) -> Result<String, String> {
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
        let prompt = gemini_prompt(prompt);
        let request_body = serde_json::json!({
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
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
            .timeout(Duration::from_secs(10))
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
    fn normalizes_flat_until_condition() {
        let normalized = normalize_gemini_response(
            r#"{"commands":[{"action":"forward","speed":50,"duration_ms":1000,"until_sensor":"distance","until_op":"lt","until_value":20}]}"#,
        )
        .expect("response should normalize");

        assert_eq!(
            normalized,
            r#"{"commands":[{"action":"forward","duration_ms":1000,"speed":50,"until":{"op":"lt","sensor":"distance","value":20}}]}"#
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
