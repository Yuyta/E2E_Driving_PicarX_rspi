use std::sync::Mutex;

use rusqlite::{Connection, params};

use super::model::{ConversationSummary, Memory, Message};

/// SQLiteを使った会話メモリの永続化
pub struct MemoryRepository {
    conn: Mutex<Connection>,
}

impl MemoryRepository {
    /// DB接続を開き、テーブルが存在しなければ作成する
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn =
            Connection::open(db_path).map_err(|e| format!("SQLite DB を開けません: {}", e))?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                summarized INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            ",
        )
        .map_err(|e| format!("テーブル作成に失敗: {}", e))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// インメモリDBで初期化 (テスト用)
    #[cfg(test)]
    pub fn new_in_memory() -> Result<Self, String> {
        Self::new(":memory:")
    }

    // ---- conversations ----

    /// 新しい会話IDを登録する
    pub fn create_conversation(&self, conversation_id: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id) VALUES (?1)",
            params![conversation_id],
        )
        .map_err(|e| format!("会話の作成に失敗: {}", e))?;
        Ok(())
    }

    // ---- messages ----

    /// メッセージを保存する
    pub fn save_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> Result<i64, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?1, ?2, ?3)",
            params![conversation_id, role, content],
        )
        .map_err(|e| format!("メッセージの保存に失敗: {}", e))?;
        Ok(conn.last_insert_rowid())
    }

    /// 直近N件のメッセージを取得する (古い順)
    pub fn get_recent_messages(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> Result<Vec<Message>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, conversation_id, role, content, created_at, summarized
                 FROM messages
                 WHERE conversation_id = ?1
                 ORDER BY id DESC
                 LIMIT ?2",
            )
            .map_err(|e| format!("クエリの準備に失敗: {}", e))?;

        let rows = stmt
            .query_map(params![conversation_id, limit as i64], |row| {
                Ok(Message {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    created_at: row.get(4)?,
                    summarized: row.get::<_, i32>(5)? != 0,
                })
            })
            .map_err(|e| format!("メッセージの取得に失敗: {}", e))?;

        let mut messages: Vec<Message> = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("メッセージの読み取りに失敗: {}", e))?;

        // DESC で取得したので古い順に並べ替え
        messages.reverse();
        Ok(messages)
    }

    /// 未要約メッセージの件数を取得する
    pub fn get_unsummarized_count(&self, conversation_id: &str) -> Result<usize, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?1 AND summarized = 0",
            params![conversation_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count as usize)
        .map_err(|e| format!("未要約件数の取得に失敗: {}", e))
    }

    /// 未要約メッセージ一覧を取得する (古い順)
    pub fn get_unsummarized_messages(&self, conversation_id: &str) -> Result<Vec<Message>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, conversation_id, role, content, created_at, summarized
                 FROM messages
                 WHERE conversation_id = ?1 AND summarized = 0
                 ORDER BY id ASC",
            )
            .map_err(|e| format!("クエリの準備に失敗: {}", e))?;

        let rows = stmt
            .query_map(params![conversation_id], |row| {
                Ok(Message {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    created_at: row.get(4)?,
                    summarized: row.get::<_, i32>(5)? != 0,
                })
            })
            .map_err(|e| format!("未要約メッセージの取得に失敗: {}", e))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("未要約メッセージの読み取りに失敗: {}", e))
    }

    /// 指定したメッセージIDを要約済みにマークする
    pub fn mark_as_summarized(&self, message_ids: &[i64]) -> Result<(), String> {
        if message_ids.is_empty() {
            return Ok(());
        }
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        let placeholders: Vec<String> = message_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "UPDATE messages SET summarized = 1 WHERE id IN ({})",
            placeholders.join(", ")
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("クエリの準備に失敗: {}", e))?;

        let params: Vec<Box<dyn rusqlite::types::ToSql>> = message_ids
            .iter()
            .map(|id| Box::new(*id) as Box<dyn rusqlite::types::ToSql>)
            .collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        stmt.execute(param_refs.as_slice())
            .map_err(|e| format!("要約済みマークに失敗: {}", e))?;
        Ok(())
    }

    // ---- conversation_summaries ----

    /// 要約を保存/更新する
    pub fn upsert_summary(&self, conversation_id: &str, summary: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        conn.execute(
            "INSERT INTO conversation_summaries (conversation_id, summary)
             VALUES (?1, ?2)
             ON CONFLICT(conversation_id) DO UPDATE SET
                 summary = excluded.summary,
                 updated_at = datetime('now')",
            params![conversation_id, summary],
        )
        .map_err(|e| format!("要約の保存に失敗: {}", e))?;
        Ok(())
    }

    /// 要約を取得する
    pub fn get_summary(
        &self,
        conversation_id: &str,
    ) -> Result<Option<ConversationSummary>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, conversation_id, summary, updated_at
                 FROM conversation_summaries
                 WHERE conversation_id = ?1",
            )
            .map_err(|e| format!("クエリの準備に失敗: {}", e))?;

        let mut rows = stmt
            .query_map(params![conversation_id], |row| {
                Ok(ConversationSummary {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    summary: row.get(2)?,
                    updated_at: row.get(3)?,
                })
            })
            .map_err(|e| format!("要約の取得に失敗: {}", e))?;

        match rows.next() {
            Some(Ok(summary)) => Ok(Some(summary)),
            Some(Err(e)) => Err(format!("要約の読み取りに失敗: {}", e)),
            None => Ok(None),
        }
    }

    // ---- memories (長期記憶) ----

    /// 長期記憶を保存/更新する
    pub fn upsert_memory(&self, key: &str, value: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        conn.execute(
            "INSERT INTO memories (key, value)
             VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET
                 value = excluded.value,
                 updated_at = datetime('now')",
            params![key, value],
        )
        .map_err(|e| format!("長期記憶の保存に失敗: {}", e))?;
        Ok(())
    }

    /// 全長期記憶を取得する
    pub fn get_all_memories(&self) -> Result<Vec<Memory>, String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        let mut stmt = conn
            .prepare("SELECT id, key, value, updated_at FROM memories ORDER BY key ASC")
            .map_err(|e| format!("クエリの準備に失敗: {}", e))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    key: row.get(1)?,
                    value: row.get(2)?,
                    updated_at: row.get(3)?,
                })
            })
            .map_err(|e| format!("長期記憶の取得に失敗: {}", e))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("長期記憶の読み取りに失敗: {}", e))
    }

    /// 長期記憶を削除する
    pub fn delete_memory(&self, key: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("DBロックの取得に失敗: {}", e))?;
        conn.execute("DELETE FROM memories WHERE key = ?1", params![key])
            .map_err(|e| format!("長期記憶の削除に失敗: {}", e))?;
        Ok(())
    }

    /// 長期記憶をJSON形式の文字列として取得する
    pub fn get_memories_as_json(&self) -> Result<String, String> {
        let memories = self.get_all_memories()?;
        if memories.is_empty() {
            return Ok("{}".to_string());
        }
        let mut map = serde_json::Map::new();
        for mem in &memories {
            // value をJSONとしてパースを試み、できなければ文字列として格納
            let json_value = serde_json::from_str(&mem.value)
                .unwrap_or_else(|_| serde_json::Value::String(mem.value.clone()));
            map.insert(mem.key.clone(), json_value);
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(map))
            .map_err(|e| format!("長期記憶のJSON変換に失敗: {}", e))
    }

    /// Geminiに送信するコンテキスト文字列を構築する
    pub fn build_context(&self, conversation_id: &str) -> Result<String, String> {
        let mut sections = Vec::new();

        // 1. Long-term Memory
        let memories_json = self.get_memories_as_json()?;
        if memories_json != "{}" {
            sections.push(format!("--- Long-term Memory ---\n{}", memories_json));
        }

        // 2. Conversation Summary
        if let Some(summary) = self.get_summary(conversation_id)? {
            sections.push(format!("--- Conversation Summary ---\n{}", summary.summary));
        }

        // 3. Recent Conversation (直近6メッセージ)
        let recent = self.get_recent_messages(conversation_id, 6)?;
        if !recent.is_empty() {
            let mut conv_lines = Vec::new();
            for msg in &recent {
                let role_label = if msg.role == "user" {
                    "User"
                } else {
                    "Assistant"
                };
                conv_lines.push(format!("[{}]: {}", role_label, msg.content));
            }
            sections.push(format!(
                "--- Recent Conversation ---\n{}",
                conv_lines.join("\n")
            ));
        }

        Ok(sections.join("\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_retrieve_messages() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.create_conversation("conv1").unwrap();

        repo.save_message("conv1", "user", "こんにちは").unwrap();
        repo.save_message(
            "conv1",
            "assistant",
            "こんにちは！何かお手伝いしましょうか？",
        )
        .unwrap();

        let messages = repo.get_recent_messages("conv1", 10).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "こんにちは");
        assert_eq!(messages[1].role, "assistant");
    }

    #[test]
    fn recent_messages_returns_limited_results_in_order() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.create_conversation("conv1").unwrap();

        for i in 1..=10 {
            repo.save_message("conv1", "user", &format!("msg{}", i))
                .unwrap();
        }

        let messages = repo.get_recent_messages("conv1", 3).unwrap();
        assert_eq!(messages.len(), 3);
        // 直近3件が古い順で返される
        assert_eq!(messages[0].content, "msg8");
        assert_eq!(messages[1].content, "msg9");
        assert_eq!(messages[2].content, "msg10");
    }

    #[test]
    fn unsummarized_count_and_marking() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.create_conversation("conv1").unwrap();

        let id1 = repo.save_message("conv1", "user", "msg1").unwrap();
        let id2 = repo.save_message("conv1", "assistant", "msg2").unwrap();
        repo.save_message("conv1", "user", "msg3").unwrap();

        assert_eq!(repo.get_unsummarized_count("conv1").unwrap(), 3);

        repo.mark_as_summarized(&[id1, id2]).unwrap();
        assert_eq!(repo.get_unsummarized_count("conv1").unwrap(), 1);
    }

    #[test]
    fn upsert_and_get_summary() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.create_conversation("conv1").unwrap();

        assert!(repo.get_summary("conv1").unwrap().is_none());

        repo.upsert_summary("conv1", "ユーザーは前進を指示した")
            .unwrap();
        let summary = repo.get_summary("conv1").unwrap().unwrap();
        assert_eq!(summary.summary, "ユーザーは前進を指示した");

        // 更新
        repo.upsert_summary("conv1", "ユーザーは前進と後退を指示した")
            .unwrap();
        let summary = repo.get_summary("conv1").unwrap().unwrap();
        assert_eq!(summary.summary, "ユーザーは前進と後退を指示した");
    }

    #[test]
    fn upsert_get_delete_memory() {
        let repo = MemoryRepository::new_in_memory().unwrap();

        repo.upsert_memory("user.name", "\"太郎\"").unwrap();
        repo.upsert_memory("robot.name", "\"ピカちゃん\"").unwrap();

        let memories = repo.get_all_memories().unwrap();
        assert_eq!(memories.len(), 2);

        // 更新
        repo.upsert_memory("user.name", "\"花子\"").unwrap();
        let memories = repo.get_all_memories().unwrap();
        assert_eq!(memories.len(), 2);
        assert_eq!(
            memories
                .iter()
                .find(|m| m.key == "user.name")
                .unwrap()
                .value,
            "\"花子\""
        );

        // 削除
        repo.delete_memory("user.name").unwrap();
        let memories = repo.get_all_memories().unwrap();
        assert_eq!(memories.len(), 1);
    }

    #[test]
    fn memories_as_json() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.upsert_memory("user.name", "\"太郎\"").unwrap();
        repo.upsert_memory("robot.default_speed", "30").unwrap();

        let json = repo.get_memories_as_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["user.name"], "太郎");
        assert_eq!(parsed["robot.default_speed"], 30);
    }

    #[test]
    fn build_context_includes_all_sections() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.create_conversation("conv1").unwrap();

        repo.upsert_memory("user.name", "\"太郎\"").unwrap();
        repo.upsert_summary("conv1", "テスト要約").unwrap();
        repo.save_message("conv1", "user", "前に進んで").unwrap();
        repo.save_message("conv1", "assistant", "はい、前進します！")
            .unwrap();

        let context = repo.build_context("conv1").unwrap();
        assert!(context.contains("Long-term Memory"));
        assert!(context.contains("太郎"));
        assert!(context.contains("Conversation Summary"));
        assert!(context.contains("テスト要約"));
        assert!(context.contains("Recent Conversation"));
        assert!(context.contains("[User]: 前に進んで"));
        assert!(context.contains("[Assistant]: はい、前進します！"));
    }

    #[test]
    fn build_context_empty_db() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.create_conversation("conv1").unwrap();

        let context = repo.build_context("conv1").unwrap();
        assert!(context.is_empty());
    }
}
