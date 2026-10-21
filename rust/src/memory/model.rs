/// DBに保存するメッセージ
#[derive(Debug, Clone)]
pub struct Message {
    pub id: i64,
    pub conversation_id: String,
    pub role: String, // "user" | "assistant"
    pub content: String,
    pub created_at: String, // ISO 8601
    pub summarized: bool,
}

/// 会話要約
#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: i64,
    pub conversation_id: String,
    pub summary: String,
    pub updated_at: String,
}

/// 長期記憶エントリ
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: i64,
    pub key: String,   // e.g. "user.name", "robot.name"
    pub value: String, // JSON value (文字列)
    pub updated_at: String,
}
