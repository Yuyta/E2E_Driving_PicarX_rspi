use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use super::common::{LlmProvider, ProviderAgent, SYSTEM_PROMPT, command_schema};

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
    let mut schema = command_schema();
    convert_schema_types_for_gemini(&mut schema);
    schema
}

fn convert_schema_types_for_gemini(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if let Some(Value::String(type_name)) = object.get_mut("type") {
                *type_name = match type_name.as_str() {
                    "object" => "OBJECT",
                    "array" => "ARRAY",
                    "string" => "STRING",
                    "integer" => "INTEGER",
                    "number" => "NUMBER",
                    "boolean" => "BOOLEAN",
                    other => other,
                }
                .to_string();
            }
            for child in object.values_mut() {
                convert_schema_types_for_gemini(child);
            }
        }
        Value::Array(values) => {
            for child in values {
                convert_schema_types_for_gemini(child);
            }
        }
        _ => {}
    }
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
                Ok(part.text.clone())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::command_response_schema;

    #[test]
    fn schema_allows_nested_condition_operators() {
        let schema = command_response_schema();
        let condition = &schema["$defs"]["Condition"];
        let properties = condition["properties"]
            .as_object()
            .expect("until properties should be an object");

        for operator in ["and", "or", "xor", "not"] {
            assert!(properties.contains_key(operator));
        }

        assert_eq!(properties["and"]["type"], "ARRAY");
        assert_eq!(properties["and"]["items"]["$ref"], "#/$defs/Condition");
        assert_eq!(
            schema["properties"]["commands"]["items"]["properties"]["until"]["$ref"],
            "#/$defs/Condition"
        );
    }
}
