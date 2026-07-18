mod command;
mod core;
pub mod gemini_agent;

pub use command::CommandQueue;

// 共通のレスポンス型
#[derive(Debug)]
pub enum LlmResponse {
    Ok(Vec<command::Command>),
    Err(String),
}

// LLMの選択肢
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmType {
    Gemini,
    Ollama,
}

pub use core::LlmAgent;
pub use gemini_agent::GeminiAgent;
