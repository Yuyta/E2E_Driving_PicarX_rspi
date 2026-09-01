use super::repository::MemoryRepository;

/// 未要約メッセージが閾値以上になったら要約を実行する
///
/// `summarize_fn` は会話テキストを受け取り、要約テキストを返すクロージャ。
/// Gemini APIの要約専用呼び出しを外部から注入する。
pub fn check_and_summarize<F>(
    repo: &MemoryRepository,
    conversation_id: &str,
    summarize_fn: F,
) -> Result<bool, String>
where
    F: FnOnce(&str) -> Result<String, String>,
{
    const SUMMARIZE_THRESHOLD: usize = 12;

    let unsummarized_count = repo.get_unsummarized_count(conversation_id)?;
    if unsummarized_count < SUMMARIZE_THRESHOLD {
        return Ok(false);
    }

    println!(
        "[Memory] 未要約メッセージが{}件 (閾値: {}) → 要約を実行します",
        unsummarized_count, SUMMARIZE_THRESHOLD
    );

    let messages = repo.get_unsummarized_messages(conversation_id)?;
    let message_ids: Vec<i64> = messages.iter().map(|m| m.id).collect();

    // 会話テキストを構築
    let conversation_text: String = messages
        .iter()
        .map(|m| {
            let role = if m.role == "user" {
                "User"
            } else {
                "Assistant"
            };
            format!("[{}]: {}", role, m.content)
        })
        .collect::<Vec<_>>()
        .join("\n");

    // 既存の要約があれば統合する
    let existing_summary = repo.get_summary(conversation_id)?;
    let prompt = match &existing_summary {
        Some(summary) => format!(
            "以下の会話を簡潔に要約してください。ユーザーの名前、ロボットの名前、\
             重要な設定変更、繰り返されるパターンなどを必ず含めてください。\n\n\
             既存の要約:\n{}\n\n\
             新しい会話:\n{}",
            summary.summary, conversation_text
        ),
        None => format!(
            "以下の会話を簡潔に要約してください。ユーザーの名前、ロボットの名前、\
             重要な設定変更、繰り返されるパターンなどを必ず含めてください。\n\n\
             会話:\n{}",
            conversation_text
        ),
    };

    let summary = summarize_fn(&prompt)?;

    // 要約を保存し、メッセージを要約済みにマークする
    repo.upsert_summary(conversation_id, &summary)?;
    repo.mark_as_summarized(&message_ids)?;

    println!(
        "[Memory] 要約完了: {}件のメッセージを要約済みにマークしました",
        message_ids.len()
    );

    Ok(true)
}

/// memory_updates のバリデーションを行う
///
/// 許可されたキーのみを受け入れ、不正なキーは無視してログ出力する。
/// 値が null の場合は削除として処理する。
pub fn validate_and_apply_memory_updates(
    repo: &MemoryRepository,
    updates: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), String> {
    for (key, value) in updates {
        if !is_allowed_key(key) {
            println!("[Memory] ⚠ 不正なメモリキーを無視: {} = {}", key, value);
            continue;
        }

        if value.is_null() {
            println!("[Memory] 長期記憶を削除: {}", key);
            repo.delete_memory(key)?;
            continue;
        }

        // 値のバリデーション
        if let Err(reason) = validate_memory_value(key, value) {
            println!(
                "[Memory] ⚠ 不正なメモリ値を無視: {} = {} (理由: {})",
                key, value, reason
            );
            continue;
        }

        let value_str = value.to_string();
        println!("[Memory] 長期記憶を保存: {} = {}", key, value_str);
        repo.upsert_memory(key, &value_str)?;
    }
    Ok(())
}

/// 許可されたメモリキーかどうかを判定する
fn is_allowed_key(key: &str) -> bool {
    // 完全一致
    let exact_keys = [
        "user.name",
        "robot.name",
        "robot.default_speed",
        "robot.max_speed",
    ];
    if exact_keys.contains(&key) {
        return true;
    }

    // プレフィックス一致
    let prefix_keys = ["user_preferences.", "behavior."];
    prefix_keys.iter().any(|prefix| key.starts_with(prefix))
}

/// メモリ値のバリデーション
fn validate_memory_value(key: &str, value: &serde_json::Value) -> Result<(), String> {
    // name 系は文字列かつ50文字以内
    if key.ends_with(".name") || key == "user.name" || key == "robot.name" {
        let s = value
            .as_str()
            .ok_or_else(|| "名前は文字列で指定してください".to_string())?;
        if s.len() > 50 {
            return Err("名前は50文字以内で指定してください".to_string());
        }
        return Ok(());
    }

    // speed 系は数値かつ 0–100 の範囲内
    if key.contains("speed") {
        let n = value
            .as_f64()
            .ok_or_else(|| "速度は数値で指定してください".to_string())?;
        if !(0.0..=100.0).contains(&n) {
            return Err("速度は 0–100 の範囲で指定してください".to_string());
        }
        return Ok(());
    }

    // その他は特にバリデーションなし
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::repository::MemoryRepository;

    #[test]
    fn allowed_keys() {
        assert!(is_allowed_key("user.name"));
        assert!(is_allowed_key("robot.name"));
        assert!(is_allowed_key("robot.default_speed"));
        assert!(is_allowed_key("robot.max_speed"));
        assert!(is_allowed_key("user_preferences.language"));
        assert!(is_allowed_key("behavior.greeting"));

        assert!(!is_allowed_key("system.prompt"));
        assert!(!is_allowed_key("api_key"));
        assert!(!is_allowed_key("random_key"));
    }

    #[test]
    fn validate_name_values() {
        assert!(validate_memory_value("user.name", &serde_json::json!("太郎")).is_ok());
        assert!(validate_memory_value("user.name", &serde_json::json!(123)).is_err());
        assert!(validate_memory_value("user.name", &serde_json::json!("a".repeat(51))).is_err());
    }

    #[test]
    fn validate_speed_values() {
        assert!(validate_memory_value("robot.default_speed", &serde_json::json!(30)).is_ok());
        assert!(validate_memory_value("robot.default_speed", &serde_json::json!(101)).is_err());
        assert!(validate_memory_value("robot.default_speed", &serde_json::json!("fast")).is_err());
    }

    #[test]
    fn apply_memory_updates_saves_and_deletes() {
        let repo = MemoryRepository::new_in_memory().unwrap();

        let mut updates = serde_json::Map::new();
        updates.insert("user.name".to_string(), serde_json::json!("太郎"));
        updates.insert("robot.name".to_string(), serde_json::json!("ピカちゃん"));

        validate_and_apply_memory_updates(&repo, &updates).unwrap();

        let memories = repo.get_all_memories().unwrap();
        assert_eq!(memories.len(), 2);

        // null で削除
        let mut delete_updates = serde_json::Map::new();
        delete_updates.insert("user.name".to_string(), serde_json::Value::Null);
        validate_and_apply_memory_updates(&repo, &delete_updates).unwrap();

        let memories = repo.get_all_memories().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].key, "robot.name");
    }

    #[test]
    fn apply_memory_updates_ignores_invalid_keys() {
        let repo = MemoryRepository::new_in_memory().unwrap();

        let mut updates = serde_json::Map::new();
        updates.insert("user.name".to_string(), serde_json::json!("太郎"));
        updates.insert("invalid_key".to_string(), serde_json::json!("bad"));

        validate_and_apply_memory_updates(&repo, &updates).unwrap();

        let memories = repo.get_all_memories().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].key, "user.name");
    }

    #[test]
    fn check_and_summarize_below_threshold() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.create_conversation("conv1").unwrap();

        for i in 0..10 {
            repo.save_message("conv1", "user", &format!("msg{}", i))
                .unwrap();
        }

        let summarized = check_and_summarize(&repo, "conv1", |_| {
            panic!("should not be called");
        })
        .unwrap();

        assert!(!summarized);
    }

    #[test]
    fn check_and_summarize_above_threshold() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.create_conversation("conv1").unwrap();

        for i in 0..12 {
            repo.save_message("conv1", "user", &format!("msg{}", i))
                .unwrap();
        }

        let summarized =
            check_and_summarize(&repo, "conv1", |_prompt| Ok("テスト要約".to_string())).unwrap();

        assert!(summarized);
        let summary = repo.get_summary("conv1").unwrap().unwrap();
        assert_eq!(summary.summary, "テスト要約");
        assert_eq!(repo.get_unsummarized_count("conv1").unwrap(), 0);
    }
}
