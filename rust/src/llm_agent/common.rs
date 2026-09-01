use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;

use serde::Deserialize;
use serde_json::{Map, Value};

use super::command::Command;

/// Both providers use the same command semantics and defaults.
pub const SYSTEM_PROMPT: &str = r#"You are a command interpreter for a PiCar-X robot car.
Given a natural-language instruction in Japanese or English, respond with a structured JSON object containing "commands" and "reply".

Supported commands:
forward(speed, duration_ms, optional angle and until),
backward(speed, duration_ms, optional angle and until),
steer(angle, duration_ms, optional until),
stop,
set_cam_pan(angle),
set_cam_tilt(angle),
get_distance,
get_grayscale,
get_battery_status.

For forward/backward, speed and duration_ms are always required.
Use speed 50 when speed is not specified.
Convert time expressions such as 5s, 10s, 5 seconds, 10 seconds, or 5秒 into duration_ms.
A time expression must never be interpreted as an until condition.
Use until only when the user explicitly requests a sensor-based condition such as "until distance is less than 10" or "距離が10未満になるまで".
For forward/backward, omit angle for straight movement. Negative angle means left and positive angle means right.
When turning while moving, use forward/backward with an angle. If the user specifies turning but not an angle, use 30 for right and -30 for left.
Execute commands in the exact order given. Return only the commands needed; do not invent extra commands.
Do not add a stop command unless explicitly requested.

An until condition is optional on movement commands. Use it when the user explicitly requests condition-based stopping, such as "stop when distance is less than 10" or "距離が10未満になるまで前進".
When until is specified, duration_ms is still required and is the maximum execution time; the command ends when the condition becomes true or duration_ms expires, whichever comes first.
Use sensor-based conditions directly; do not add a separate get_distance command.

The "reply" must be short, natural, and in the same language as the user's instruction.



Examples:
"back for 5s" → backward, speed 50, duration_ms 5000.
"Go straight for 10s" → forward, speed 50, duration_ms 10000.
"右に曲がりながら5秒進んで、そのあと3秒まっすぐ進んでください" → forward with angle 30 for 5000ms, then forward without angle for 3000ms.
"距離が10未満になるまで前進してください" → forward with until distance lt 10.

Do not output prose or markdown outside the structured response.

Memory and Personalization Rules:
- If a "Long-term Memory" section is provided in the context, use names and preferences stored there (e.g. user's name, robot's name, default speed).
- If the user names you or changes your name (e.g. "あなたの名前は太郎です", "これからはポチと呼ぶね", "名前をピカにして"), treat this as YOUR (the robot's) name and set "robot.name" in memory_updates. Acknowledge and adopt your new name in your reply.
- If the user tells you their own name (e.g. "私の名前は太郎です", "太郎と呼んで"), set "user.name" in memory_updates.
- memory_updates format: { "key": "value" } where key uses dot notation (e.g. "user.name", "robot.name", "robot.default_speed").
- To delete a memory, set its value to null.
- Only include memory_updates when the user provides new persistent information."#;

/// Common response type consumed by the application loop.
#[derive(Debug)]
pub enum LlmResponse {
    Ok {
        request_id: u64,
        commands: Vec<Command>,
        reply: String,
        memory_updates: Option<Map<String, Value>>,
    },
    Err {
        request_id: u64,
        message: String,
    },
}

impl LlmResponse {
    pub fn request_id(&self) -> u64 {
        match self {
            Self::Ok { request_id, .. } | Self::Err { request_id, .. } => *request_id,
        }
    }
}

/// Provider-specific code only needs to return the command JSON text.
/// Transport, parsing, and background execution are shared by all providers.
pub trait LlmProvider: Send + 'static {
    fn request(&self, prompt: &str) -> Result<String, String>;
    fn request_audio(&self, _wav_data: &[u8]) -> Result<String, String> {
        Err("このLLMプロバイダは音声入力に対応していません".to_string())
    }

    /// メモリコンテキスト付きリクエスト（デフォルトはコンテキスト無視でrequest呼び出し）
    fn request_with_context(&self, prompt: &str, _context: &str) -> Result<String, String> {
        self.request(prompt)
    }

    /// メモリコンテキスト付き音声リクエスト（デフォルトはコンテキスト無視でrequest_audio呼び出し）
    fn request_audio_with_context(
        &self,
        wav_data: &[u8],
        _context: &str,
    ) -> Result<String, String> {
        self.request_audio(wav_data)
    }

    /// テキスト要約用のリクエスト (コマンドスキーマなし、テキスト生成のみ)
    /// デフォルトは未実装
    fn request_summarize(&self, _prompt: &str) -> Result<String, String> {
        Err("このLLMプロバイダは要約機能に対応していません".to_string())
    }
}

/// Non-blocking LLM agent shared by every provider.
pub struct ProviderAgent<P> {
    prompt_tx: Sender<Prompt>,
    response_rx: Receiver<LlmResponse>,
    next_request_id: AtomicU64,
    cancelled_through: Arc<AtomicU64>,
    provider: PhantomData<P>,
}

enum PromptContent {
    Text(String),
    Audio(Vec<u8>),
}

struct Prompt {
    request_id: u64,
    content: PromptContent,
    /// メモリコンテキスト (長期記憶・要約・直近会話)
    context: Option<String>,
}

impl<P> ProviderAgent<P>
where
    P: LlmProvider + Default,
{
    pub fn new() -> Self {
        Self::with_provider(P::default())
    }
}

impl<P> ProviderAgent<P>
where
    P: LlmProvider,
{
    pub fn with_provider(provider: P) -> Self {
        let (prompt_tx, prompt_rx) = mpsc::channel::<Prompt>();
        let (response_tx, response_rx) = mpsc::channel::<LlmResponse>();
        let cancelled_through = Arc::new(AtomicU64::new(0));
        let worker_cancelled_through = Arc::clone(&cancelled_through);

        thread::spawn(move || {
            for prompt in prompt_rx {
                if is_cancelled(&worker_cancelled_through, prompt.request_id) {
                    continue;
                }

                let response = match (&prompt.content, &prompt.context) {
                    (PromptContent::Text(text), Some(ctx)) => match provider
                        .request_with_context(text, ctx)
                        .and_then(|json| _parse_commands_for_prompt(&json, text))
                    {
                        Ok((commands, reply, memory_updates)) => LlmResponse::Ok {
                            request_id: prompt.request_id,
                            commands,
                            reply,
                            memory_updates,
                        },
                        Err(message) => LlmResponse::Err {
                            request_id: prompt.request_id,
                            message,
                        },
                    },
                    (PromptContent::Text(text), None) => match provider
                        .request(text)
                        .and_then(|json| _parse_commands_for_prompt(&json, text))
                    {
                        Ok((commands, reply, memory_updates)) => LlmResponse::Ok {
                            request_id: prompt.request_id,
                            commands,
                            reply,
                            memory_updates,
                        },
                        Err(message) => LlmResponse::Err {
                            request_id: prompt.request_id,
                            message,
                        },
                    },
                    (PromptContent::Audio(wav_data), Some(ctx)) => match provider
                        .request_audio_with_context(wav_data, ctx)
                        .and_then(|json| _parse_commands_for_prompt(&json, ""))
                    {
                        Ok((commands, reply, memory_updates)) => LlmResponse::Ok {
                            request_id: prompt.request_id,
                            commands,
                            reply,
                            memory_updates,
                        },
                        Err(message) => LlmResponse::Err {
                            request_id: prompt.request_id,
                            message,
                        },
                    },
                    (PromptContent::Audio(wav_data), None) => match provider
                        .request_audio(wav_data)
                        .and_then(|json| _parse_commands_for_prompt(&json, ""))
                    {
                        Ok((commands, reply, memory_updates)) => LlmResponse::Ok {
                            request_id: prompt.request_id,
                            commands,
                            reply,
                            memory_updates,
                        },
                        Err(message) => LlmResponse::Err {
                            request_id: prompt.request_id,
                            message,
                        },
                    },
                };

                if is_cancelled(&worker_cancelled_through, prompt.request_id) {
                    continue;
                }

                if response_tx.send(response).is_err() {
                    break;
                }
            }
        });

        Self {
            prompt_tx,
            response_rx,
            next_request_id: AtomicU64::new(1),
            cancelled_through,
            provider: PhantomData,
        }
    }

    /// Submit a text prompt without blocking the caller and return its request ID.
    pub fn submit_prompt(&self, text: String) -> Result<u64, String> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.prompt_tx
            .send(Prompt {
                request_id,
                content: PromptContent::Text(text),
                context: None,
            })
            .map(|_| request_id)
            .map_err(|_| "LLM workerが停止しているため要求を登録できません".to_string())
    }

    /// Submit a text prompt with memory context.
    pub fn submit_prompt_with_context(&self, text: String, context: String) -> Result<u64, String> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.prompt_tx
            .send(Prompt {
                request_id,
                content: PromptContent::Text(text),
                context: Some(context),
            })
            .map(|_| request_id)
            .map_err(|_| "LLM workerが停止しているため要求を登録できません".to_string())
    }

    /// Submit an audio prompt without blocking the caller and return its request ID.
    pub fn submit_audio(&self, wav_data: Vec<u8>) -> Result<u64, String> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.prompt_tx
            .send(Prompt {
                request_id,
                content: PromptContent::Audio(wav_data),
                context: None,
            })
            .map(|_| request_id)
            .map_err(|_| "LLM workerが停止しているため要求を登録できません".to_string())
    }

    /// Submit an audio prompt with memory context.
    pub fn submit_audio_with_context(
        &self,
        wav_data: Vec<u8>,
        context: String,
    ) -> Result<u64, String> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.prompt_tx
            .send(Prompt {
                request_id,
                content: PromptContent::Audio(wav_data),
                context: Some(context),
            })
            .map(|_| request_id)
            .map_err(|_| "LLM workerが停止しているため要求を登録できません".to_string())
    }

    /// Cancel this request and all older requests.
    ///
    /// A provider's in-flight HTTP call cannot be interrupted through the
    /// common trait, but queued work and responses produced after cancellation
    /// are suppressed.
    pub fn cancel(&self, request_id: u64) {
        advance_atomic_max(&self.cancelled_through, request_id);
    }

    /// Receive a response if one is already available.
    pub fn try_recv_response(&self) -> Result<Option<LlmResponse>, String> {
        match self.response_rx.try_recv() {
            Ok(response) => Ok(Some(response)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err("LLM workerが停止したため応答を受信できません".to_string())
            }
        }
    }
}

fn is_cancelled(cancelled_through: &AtomicU64, request_id: u64) -> bool {
    request_id <= cancelled_through.load(Ordering::Relaxed)
}

fn advance_atomic_max(value: &AtomicU64, candidate: u64) {
    let mut current = value.load(Ordering::Relaxed);
    while current < candidate {
        match value.compare_exchange_weak(current, candidate, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

#[derive(Deserialize)]
struct CommandListPayload {
    commands: Vec<Command>,
    reply: String,
    #[serde(default)]
    memory_updates: Option<Map<String, Value>>,
}

/// Standard JSON Schema shared by both provider transports.
pub fn command_schema() -> Value {
    serde_json::json!({
        "$defs": {
            "Condition": {
                "type": "object",
                "properties": {
                    "sensor": {
                        "type": "string",
                        "enum": [
                            "distance", "grayscale_left", "grayscale_center",
                            "grayscale_right", "battery_voltage", "battery_percent"
                        ]
                    },
                    "op": {
                        "type": "string",
                        "enum": ["eq", "ne", "gte", "lte", "gt", "lt"]
                    },
                    "value": { "type": "number" },
                    "and": {
                        "type": "array",
                        "minItems": 1,
                        "items": { "$ref": "#/$defs/Condition" }
                    },
                    "or": {
                        "type": "array",
                        "minItems": 1,
                        "items": { "$ref": "#/$defs/Condition" }
                    },
                    "xor": {
                        "type": "array",
                        "minItems": 1,
                        "items": { "$ref": "#/$defs/Condition" }
                    },
                    "not": { "$ref": "#/$defs/Condition" }
                },
                "additionalProperties": false
            }
        },
        "type": "object",
        "properties": {
            "commands": {
                "type": "array",
                "description": "List of robot commands to execute in sequence.",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "forward", "backward", "steer", "stop",
                                "set_cam_pan", "set_cam_tilt", "get_distance",
                                "get_grayscale", "get_battery_status"
                            ]
                        },
                        "speed": { "type": "integer", "minimum": 0, "maximum": 100 },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 30000,
                            "description": "Movement duration in milliseconds. Use this for time-based requests such as 10 seconds, 10s, 10000ms, or 10秒."
                        },
                        "angle": { "type": "integer", "minimum": -90, "maximum": 90 },
                        "until": { "$ref": "#/$defs/Condition" }
                    },
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "action": { "enum": ["forward", "backward"] }
                            },
                            "required": ["action", "speed", "duration_ms"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": { "enum": ["steer"] }
                            },
                            "required": ["action", "angle", "duration_ms"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": { "enum": ["set_cam_pan", "set_cam_tilt"] }
                            },
                            "required": ["action", "angle"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "enum": [
                                        "stop", "get_distance", "get_grayscale",
                                        "get_battery_status"
                                    ]
                                }
                            },
                            "required": ["action"]
                        }
                    ],
                    "required": ["action"],
                    "additionalProperties": false
                }
            },
            "reply": {
                "type": "string",
                "description": "A short, natural reply message in Japanese to the user's request."
            }
        },
        "required": ["commands", "reply"],
        "additionalProperties": false
    })
}

const DEFAULT_SPEED: i64 = 30;
const DEFAULT_DURATION_MS: u64 = 1_000;
const MIN_DURATION_MS: u64 = 100;
const MAX_DURATION_MS: u64 = 30_000;
const MIN_SPEED: i64 = 0;
const MAX_SPEED: i64 = 100;
const MIN_ANGLE: i64 = -90;
const MAX_ANGLE: i64 = 90;
const MAX_CONDITION_DEPTH: usize = 8;

/// Parse and normalize the provider-independent command response.
pub fn _parse_commands(
    response_text: &str,
) -> Result<(Vec<Command>, String, Option<Map<String, Value>>), String> {
    _parse_commands_for_prompt(response_text, "")
}

/// Parse and normalize the provider-independent command response using the
/// original user prompt for deterministic safety corrections.
pub fn _parse_commands_for_prompt(
    response_text: &str,
    user_prompt: &str,
) -> Result<(Vec<Command>, String, Option<Map<String, Value>>), String> {
    let mut value: Value = serde_json::from_str(response_text).map_err(|e| {
        format!(
            "LLMが返したコマンドJSONの解析に失敗: {} (受信内容: {})",
            e, response_text
        )
    })?;

    let commands = value
        .get_mut("commands")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            format!(
                "LLMが返したコマンドJSONに commands 配列がありません (受信内容: {})",
                response_text
            )
        })?;

    normalize_commands_for_user_prompt(commands, user_prompt);

    for (index, command) in commands.iter_mut().enumerate() {
        normalize_and_validate_command(command, index).map_err(|e| {
            format!(
                "LLMが返したコマンドJSONの安全値検証に失敗: {} (受信内容: {})",
                e, response_text
            )
        })?;
    }

    serde_json::from_value::<CommandListPayload>(value)
        .map(|payload| (payload.commands, payload.reply, payload.memory_updates))
        .map_err(|e| {
            format!(
                "LLMが返したコマンドJSONの解析に失敗: {} (受信内容: {})",
                e, response_text
            )
        })
}

fn normalize_commands_for_user_prompt(commands: &mut [Value], user_prompt: &str) {
    let should_strip_until =
        !user_prompt.trim().is_empty() && !prompt_requests_sensor_stop(user_prompt);

    for command in commands {
        let Some(object) = command.as_object_mut() else {
            continue;
        };
        let Some(action) = object.get("action").and_then(Value::as_str) else {
            continue;
        };
        let is_timed_movement = matches!(action, "forward" | "backward" | "steer");
        if !is_timed_movement {
            continue;
        }

        if should_strip_until {
            object.remove("until");
        }
    }
}

fn prompt_requests_sensor_stop(prompt: &str) -> bool {
    let prompt = prompt.to_lowercase();
    let sensor_words = [
        "sensor",
        "distance",
        "grayscale",
        "battery",
        "obstacle",
        "センサー",
        "距離",
        "障害物",
        "バッテリー",
        "グレースケール",
        "白線",
        "黒線",
    ];
    let stop_words = [
        "until",
        "when",
        "stop",
        "less than",
        "greater than",
        "まで",
        "なったら",
        "以下",
        "未満",
        "以上",
        "超え",
        "止",
        "停止",
    ];

    sensor_words.iter().any(|word| prompt.contains(word))
        && stop_words.iter().any(|word| prompt.contains(word))
}

fn normalize_and_validate_command(command: &mut Value, index: usize) -> Result<(), String> {
    let object = command
        .as_object_mut()
        .ok_or_else(|| format!("コマンド[{}]がオブジェクトではありません", index))?;
    let action = object
        .get("action")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("コマンド[{}]に action がありません", index))?;
    validate_allowed_fields(object, action.as_str(), index)?;

    // Some local models serialize an integer-valued JSON number as `1000.0`.
    // Converting only finite, integral values preserves the range checks below.
    for field in ["speed", "duration_ms", "angle"] {
        normalize_integral_number(object, field);
    }

    match action.as_str() {
        "forward" | "backward" => {
            set_default_u64(object, "duration_ms", DEFAULT_DURATION_MS);
            set_default_i64(object, "speed", DEFAULT_SPEED);
            validate_u64_range(
                object,
                "duration_ms",
                MIN_DURATION_MS,
                MAX_DURATION_MS,
                index,
            )?;
            validate_i64_range(object, "speed", MIN_SPEED, MAX_SPEED, index)?;
            validate_optional_i64_range(object, "angle", MIN_ANGLE, MAX_ANGLE, index)?;
            validate_until(object, index)?;
        }
        "steer" => {
            set_default_u64(object, "duration_ms", DEFAULT_DURATION_MS);
            set_default_i64(object, "angle", 0);
            validate_u64_range(
                object,
                "duration_ms",
                MIN_DURATION_MS,
                MAX_DURATION_MS,
                index,
            )?;
            validate_i64_range(object, "angle", MIN_ANGLE, MAX_ANGLE, index)?;
            validate_until(object, index)?;
        }
        "set_cam_pan" | "set_cam_tilt" => {
            set_default_i64(object, "angle", 0);
            validate_i64_range(object, "angle", MIN_ANGLE, MAX_ANGLE, index)?;
        }
        "stop" | "get_distance" | "get_grayscale" | "get_battery_status" => {}
        _ => {}
    }

    Ok(())
}

fn validate_allowed_fields(
    object: &Map<String, Value>,
    action: &str,
    index: usize,
) -> Result<(), String> {
    let allowed: &[&str] = match action {
        "forward" | "backward" => &["action", "speed", "duration_ms", "angle", "until"],
        "steer" => &["action", "angle", "duration_ms", "until"],
        "set_cam_pan" | "set_cam_tilt" => &["action", "angle"],
        "stop" | "get_distance" | "get_grayscale" | "get_battery_status" => &["action"],
        _ => return Ok(()),
    };

    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(format!(
            "コマンド[{}]に未対応フィールド {} が含まれています",
            index, field
        ));
    }
    Ok(())
}

fn set_default_u64(object: &mut Map<String, Value>, field: &str, default: u64) {
    if object.get(field).is_none_or(Value::is_null) {
        object.insert(field.to_string(), Value::from(default));
    }
}

fn set_default_i64(object: &mut Map<String, Value>, field: &str, default: i64) {
    if object.get(field).is_none_or(Value::is_null) {
        object.insert(field.to_string(), Value::from(default));
    }
}

fn normalize_integral_number(object: &mut Map<String, Value>, field: &str) {
    let Some(value) = object.get(field).cloned() else {
        return;
    };
    if value.as_i64().is_some() || value.as_u64().is_some() {
        return;
    }

    let Some(number) = value.as_f64() else {
        return;
    };
    if !number.is_finite() || number.fract() != 0.0 || number < 0.0 {
        if field == "angle" && number.is_finite() && number.fract() == 0.0 {
            object.insert(field.to_string(), Value::from(number as i64));
        }
        return;
    }

    object.insert(field.to_string(), Value::from(number as u64));
}

fn validate_u64_range(
    object: &Map<String, Value>,
    field: &str,
    min: u64,
    max: u64,
    index: usize,
) -> Result<(), String> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("コマンド[{}]の {} は整数で指定してください", index, field))?;
    if !(min..=max).contains(&value) {
        return Err(format!(
            "コマンド[{}]の {}={} は {}..{} の範囲外です",
            index, field, value, min, max
        ));
    }
    Ok(())
}

fn validate_i64_range(
    object: &Map<String, Value>,
    field: &str,
    min: i64,
    max: i64,
    index: usize,
) -> Result<(), String> {
    let value = object
        .get(field)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("コマンド[{}]の {} は整数で指定してください", index, field))?;
    if !(min..=max).contains(&value) {
        return Err(format!(
            "コマンド[{}]の {}={} は {}..{} の範囲外です",
            index, field, value, min, max
        ));
    }
    Ok(())
}

fn validate_optional_i64_range(
    object: &Map<String, Value>,
    field: &str,
    min: i64,
    max: i64,
    index: usize,
) -> Result<(), String> {
    if object.get(field).is_some_and(|value| !value.is_null()) {
        validate_i64_range(object, field, min, max, index)?;
    }
    Ok(())
}

fn validate_until(object: &Map<String, Value>, index: usize) -> Result<(), String> {
    let Some(until) = object.get("until") else {
        return Ok(());
    };
    if until.is_null() {
        return Ok(());
    }
    validate_condition(until, index, 0)
}

fn validate_condition(value: &Value, command_index: usize, depth: usize) -> Result<(), String> {
    if depth > MAX_CONDITION_DEPTH {
        return Err(format!(
            "コマンド[{}]の until 条件が深すぎます（最大{}階層）",
            command_index, MAX_CONDITION_DEPTH
        ));
    }

    let object = value.as_object().ok_or_else(|| {
        format!(
            "コマンド[{}]の until 条件はオブジェクトで指定してください",
            command_index
        )
    })?;

    let logical_keys = ["and", "or", "xor", "not"]
        .into_iter()
        .filter(|key| object.contains_key(*key))
        .collect::<Vec<_>>();
    let is_leaf = ["sensor", "op", "value"]
        .into_iter()
        .all(|key| object.contains_key(key));

    if logical_keys.len() > 1 || (is_leaf && !logical_keys.is_empty()) {
        return Err(format!(
            "コマンド[{}]の until 条件の形式が不正です",
            command_index
        ));
    }

    if is_leaf {
        if object.get("sensor").and_then(Value::as_str).is_none()
            || object.get("op").and_then(Value::as_str).is_none()
            || object
                .get("value")
                .and_then(Value::as_f64)
                .is_none_or(|value| !value.is_finite())
        {
            return Err(format!(
                "コマンド[{}]の until 条件の値が不正です",
                command_index
            ));
        }
        return Ok(());
    }

    let Some(logical_key) = logical_keys.first() else {
        return Err(format!(
            "コマンド[{}]の until 条件に sensor/op/value または論理演算子がありません",
            command_index
        ));
    };

    match *logical_key {
        "not" => validate_condition(
            object.get("not").expect("logical key must exist"),
            command_index,
            depth + 1,
        ),
        "and" | "or" | "xor" => {
            let children = object
                .get(*logical_key)
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    format!(
                        "コマンド[{}]の until.{} は配列で指定してください",
                        command_index, logical_key
                    )
                })?;
            if children.is_empty() {
                return Err(format!(
                    "コマンド[{}]の until.{} は1件以上指定してください",
                    command_index, logical_key
                ));
            }
            for child in children {
                validate_condition(child, command_index, depth + 1)?;
            }
            Ok(())
        }
        _ => unreachable!("logical key list contains only known keys"),
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{
        _parse_commands, _parse_commands_for_prompt, LlmProvider, LlmResponse, ProviderAgent,
    };
    use crate::llm_agent::command::Command;

    struct MockProvider;

    impl LlmProvider for MockProvider {
        fn request(&self, prompt: &str) -> Result<String, String> {
            if prompt == "error" {
                Err("mock error".to_string())
            } else {
                Ok(r#"{"commands":[{"action":"stop"}]}"#.to_string())
            }
        }
    }

    struct SlowProvider;

    impl LlmProvider for SlowProvider {
        fn request(&self, _prompt: &str) -> Result<String, String> {
            std::thread::sleep(Duration::from_millis(20));
            Ok(r#"{"commands":[{"action":"stop"}]}"#.to_string())
        }
    }

    fn receive_response(agent: &ProviderAgent<MockProvider>) -> LlmResponse {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(response) = agent
                .try_recv_response()
                .expect("mock worker should remain connected")
            {
                return response;
            }
            assert!(Instant::now() < deadline, "mock response timed out");
            std::thread::yield_now();
        }
    }

    #[test]
    fn shared_schema_uses_standard_json_schema_types() {
        let schema = super::command_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(
            schema["properties"]["commands"]["items"]["properties"]["until"]["$ref"],
            "#/$defs/Condition"
        );
        assert!(schema["properties"]["commands"]["items"]["anyOf"].is_array());
        assert_eq!(schema["$defs"]["Condition"]["type"], "object");
    }

    #[test]
    fn movement_duration_is_filled_for_all_providers() {
        let (commands, reply, _memory_updates) =
            _parse_commands(r#"{"commands":[{"action":"forward","speed":30,"duration_ms":1000},{"action":"stop"}],"reply":"ok"}"#)
                .expect("command JSON should be valid");

        assert_eq!(reply, "ok");
        assert!(matches!(
            commands.first(),
            Some(Command::Forward {
                duration_ms: 1000,
                ..
            })
        ));
        assert!(matches!(commands.get(1), Some(Command::Stop)));
    }

    #[test]
    fn parser_applies_safe_defaults_and_rejects_unsafe_values() {
        let (commands, reply, _memory_updates) = _parse_commands(
            r#"{"commands":[{"action":"forward","speed":30,"duration_ms":1000}],"reply":"ok"}"#,
        )
        .expect("safe defaults should be applied");

        assert_eq!(reply, "ok");
        match commands.first() {
            Some(Command::Forward {
                speed, duration_ms, ..
            }) => {
                assert_eq!(*speed, 30);
                assert_eq!(*duration_ms, 1000);
            }
            _ => panic!("expected a forward command"),
        }

        assert!(
            _parse_commands(
                r#"{"commands":[{"action":"forward","speed":30,"duration_ms":30001}],"reply":"ok"}"#
            )
            .is_err()
        );
        assert!(
            _parse_commands(
                r#"{"commands":[{"action":"forward","speed":101,"duration_ms":1000}],"reply":"ok"}"#
            )
            .is_err()
        );
        assert!(
            _parse_commands(r#"{"commands":[{"action":"set_cam_pan","angle":91}],"reply":"ok"}"#)
                .is_err()
        );
    }

    #[test]
    fn parser_defaults_missing_angles_and_normalizes_integral_floats() {
        let (commands, reply, _memory_updates) = _parse_commands(
            r#"{"commands":[{"action":"steer","angle":0,"duration_ms":1000.0},{"action":"set_cam_pan","angle":0},{"action":"forward","speed":40.0,"duration_ms":500.0}],"reply":"ok"}"#,
        )
        .expect("safe angle defaults and integral floats should be accepted");

        assert_eq!(reply, "ok");
        assert!(matches!(
            commands.first(),
            Some(Command::Steer {
                angle: 0,
                duration_ms: 1000,
                ..
            })
        ));
        assert!(matches!(
            commands.get(1),
            Some(Command::SetCamPan { angle: 0 })
        ));
        assert!(matches!(
            commands.get(2),
            Some(Command::Forward {
                speed: 40,
                duration_ms: 500,
                ..
            })
        ));
    }

    #[test]
    fn parser_accepts_nested_until_conditions() {
        let (commands, reply, _memory_updates) = _parse_commands(
            r#"{"commands":[{"action":"forward","speed":20,"duration_ms":500,"until":{"and":[{"sensor":"distance","op":"lt","value":10},{"not":{"sensor":"battery_percent","op":"lt","value":10}}]}}],"reply":"ok"}"#,
        )
        .expect("nested until conditions should be valid");

        assert_eq!(reply, "ok");
        assert!(matches!(commands.first(), Some(Command::Forward { .. })));
    }

    #[test]
    fn parser_rejects_invalid_until_conditions() {
        assert!(
            _parse_commands(r#"{"commands":[{"action":"forward","speed":30,"duration_ms":1000,"until":{"and":[]}}],"reply":"err"}"#).is_err()
        );
        assert!(_parse_commands(
            r#"{"commands":[{"action":"forward","speed":30,"duration_ms":1000,"until":{"and":[{"sensor":"distance","op":"lt","value":10}],"not":{"sensor":"distance","op":"lt","value":5}}}],"reply":"err"}"#
        )
        .is_err());
    }

    #[test]
    fn parser_rejects_unknown_fields_instead_of_ignoring_them() {
        assert!(
            _parse_commands(r#"{"commands":[{"action":"forward","speed":50,"until_sensor":"distance","until_value":10000}],"reply":"err"}"#)
                .is_err()
        );
    }

    #[test]
    fn prompt_duration_overrides_invented_until_condition() {
        let (commands, _, _memory_updates) = _parse_commands_for_prompt(
            r#"{"commands":[{"action":"forward","speed":50,"until":{"sensor":"distance","op":"lt","value":10}}],"reply":"ok"}"#,
            "まっすぐ10秒進んで",
        )
        .expect("time prompt should normalize to duration_ms");

        assert!(matches!(
            commands.first(),
            Some(Command::Forward {
                duration_ms: 10_000,
                until: None,
                ..
            })
        ));
    }

    #[test]
    fn prompt_duration_accepts_seconds_and_milliseconds_variants() {
        for (prompt, expected) in [
            ("go forward for 10s", 10_000),
            ("go forward for 10 seconds", 10_000),
            ("10000msだけ前進", 10_000),
            ("1.5秒進んで", 1_500),
            ("２秒前進", 2_000),
        ] {
            let (commands, _, _memory_updates) = _parse_commands_for_prompt(
                r#"{"commands":[{"action":"forward","speed":50,"duration_ms":1000}],"reply":"ok"}"#,
                prompt,
            )
            .expect("duration variants should normalize");

            assert!(matches!(
                commands.first(),
                Some(Command::Forward { duration_ms, .. }) if *duration_ms == expected
            ));
        }
    }

    #[test]
    fn prompt_sensor_stop_keeps_until_condition() {
        let (commands, _, _memory_updates) = _parse_commands_for_prompt(
            r#"{"commands":[{"action":"forward","speed":50,"duration_ms":10000,"until":{"sensor":"distance","op":"lt","value":10}}],"reply":"ok"}"#,
            "距離が10cm未満になったら止まるまで前進",
        )
        .expect("sensor prompt should keep until condition");

        assert!(matches!(
            commands.first(),
            Some(Command::Forward {
                duration_ms: 10_000,
                until: Some(_),
                ..
            })
        ));
    }

    #[test]
    fn parser_extracts_memory_updates() {
        let (_commands, reply, memory_updates) = _parse_commands(
            r#"{"commands":[{"action":"stop"}],"reply":"覚えました！","memory_updates":{"user.name":"太郎","robot.name":"ピカ"}}"#,
        )
        .expect("should parse memory updates");

        assert_eq!(reply, "覚えました！");
        let updates = memory_updates.expect("memory_updates should be present");
        assert_eq!(
            updates.get("user.name").and_then(|v| v.as_str()),
            Some("太郎")
        );
        assert_eq!(
            updates.get("robot.name").and_then(|v| v.as_str()),
            Some("ピカ")
        );
    }

    #[test]
    fn responses_keep_monotonic_request_ids() {
        let agent = ProviderAgent::with_provider(MockProvider);
        assert_eq!(agent.submit_prompt("ok".to_string()).unwrap(), 1);
        assert_eq!(agent.submit_prompt("error".to_string()).unwrap(), 2);
        assert_eq!(agent.submit_audio(vec![0, 1, 2]).unwrap(), 3);

        let first = receive_response(&agent);
        let second = receive_response(&agent);
        let third = receive_response(&agent);
        assert!(matches!(first, LlmResponse::Ok { request_id: 1, .. }));
        assert!(matches!(second, LlmResponse::Err { request_id: 2, .. }));
        assert!(matches!(third, LlmResponse::Ok { request_id: 3, .. }));
    }

    #[test]
    fn cancellation_suppresses_in_flight_response() {
        let agent = ProviderAgent::with_provider(SlowProvider);
        let request_id = agent.submit_prompt("slow".to_string()).unwrap();
        agent.cancel(request_id);

        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            assert!(
                agent.try_recv_response().unwrap().is_none(),
                "cancelled response must not reach the application"
            );
            if Instant::now() >= deadline {
                break;
            }
            std::thread::yield_now();
        }
    }
}
