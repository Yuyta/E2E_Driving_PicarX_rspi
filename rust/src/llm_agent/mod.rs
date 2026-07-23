mod command;
mod common;
mod gemini;
mod ollama;

pub use command::CommandQueue;
pub use common::LlmResponse;
pub use gemini::GeminiAgent;
pub use ollama::OllamaAgent;

// LLMの選択肢
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmType {
    Gemini,
    Ollama,
}
