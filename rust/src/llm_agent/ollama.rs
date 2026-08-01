use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::{LlmProvider, ProviderAgent, SYSTEM_PROMPT, command_schema};

const DEFAULT_OLLAMA_HOST: &str = "127.0.0.1";
const DEFAULT_OLLAMA_PORT: &str = "11434";
const DEFAULT_OLLAMA_MODEL: &str = "llama3.1";

#[derive(Serialize)]
struct OllamaMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage<'a>>,
    format: Value,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: u32,
}

#[derive(Deserialize)]
struct OllamaChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

/// Ollama-specific configuration and HTTP transport.
pub struct OllamaProvider {
    url: String,
    model: String,
}

/// Ollama agent with the same non-blocking API as every other LLM agent.
pub type OllamaAgent = ProviderAgent<OllamaProvider>;

impl Default for OllamaProvider {
    fn default() -> Self {
        let host = std::env::var("OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_HOST.to_string());
        let url = Self::build_chat_url(&host);
        let model =
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.to_string());

        println!("[LLM] Ollama接続先: {} (model={})", url, model);
        Self { url, model }
    }
}

impl OllamaProvider {
    /// Normalize a host, host:port, or complete URL to Ollama's chat endpoint.
    fn build_chat_url(input: &str) -> String {
        let mut url = input.trim().to_string();
        if url.is_empty() {
            url = DEFAULT_OLLAMA_HOST.to_string();
        }
        if !url.contains("://") {
            url = format!("http://{}", url);
        }

        // A path means the caller supplied a complete endpoint.
        let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or("");
        if after_scheme.contains('/') {
            return url;
        }

        if !after_scheme.contains(':') {
            url = format!("{}:{}", url, DEFAULT_OLLAMA_PORT);
        }

        format!("{}/api/chat", url)
    }
}

impl LlmProvider for OllamaProvider {
    fn request(&self, prompt: &str) -> Result<String, String> {
        let request_body = OllamaChatRequest {
            model: &self.model,
            messages: vec![
                OllamaMessage {
                    role: "system",
                    content: SYSTEM_PROMPT,
                },
                OllamaMessage {
                    role: "user",
                    content: prompt,
                },
            ],
            format: command_schema(),
            stream: false,
            options: OllamaOptions {
                temperature: 0.0,
                num_predict: 512,
            },
        };

        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(30))
            .build()
            .post(&self.url)
            .send_json(&request_body)
            .map_err(|e| format!("Ollamaへのリクエストに失敗: {}", e))?
            .into_json::<OllamaChatResponse>()
            .map(|response| response.message.content)
            .map_err(|e| format!("Ollamaのレスポンス解析に失敗: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::{OllamaChatRequest, OllamaOptions, OllamaProvider};
    use crate::llm_agent::common::command_schema;

    #[test]
    fn build_chat_url_accepts_host_and_complete_url() {
        assert_eq!(
            OllamaProvider::build_chat_url("192.168.0.2"),
            "http://192.168.0.2:11434/api/chat"
        );
        assert_eq!(
            OllamaProvider::build_chat_url("http://localhost:1234/api/chat"),
            "http://localhost:1234/api/chat"
        );
    }

    #[test]
    fn request_uses_command_schema_format() {
        let request = OllamaChatRequest {
            model: "test",
            messages: Vec::new(),
            format: command_schema(),
            stream: false,
            options: OllamaOptions {
                temperature: 0.0,
                num_predict: 512,
            },
        };
        let body = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(body["format"]["type"], "object");
        assert_eq!(
            body["format"]["properties"]["commands"]["items"]["properties"]["until"]["$ref"],
            "#/$defs/Condition"
        );
        assert_eq!(body["options"]["temperature"], 0.0);
        assert_eq!(body["options"]["num_predict"], 512);
    }
}
