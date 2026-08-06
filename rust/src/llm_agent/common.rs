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
Given a natural-language instruction in Japanese or English, respond with ONLY a JSON object, no prose or markdown.
The object must contain:
1. "commands": an array of command objects.
2. "reply": a short, natural Japanese message responding to the user's instruction (e.g. "はい、前進します", "こんにちは！何かお手伝いできることはありますか？").
Supported commands: forward(speed 0-100, duration_ms, optional angle and until),
backward(speed 0-100, duration_ms, optional angle and until),
steer(angle -90..90, duration_ms, optional until), stop,
set_cam_pan(angle -90..90), set_cam_tilt(angle -90..90), get_distance,
get_grayscale, and get_battery_status.
Movement commands require duration_ms as a safety-bound maximum; use 1000ms when no duration is specified.
Convert durations such as 2 seconds or 2秒 into milliseconds.
Always include speed and duration_ms for forward/backward, angle and duration_ms for steer,
and angle for set_cam_pan/set_cam_tilt. For stop and sensor-read commands, include only action.
Return only the commands needed for the user's request; do not invent extra movement or camera commands.
An until condition is optional on movement commands. Do not include any until conditions unless the user explicitly requests sensor-based stopping (e.g. "stop when distance < 10").
When explicitly requested, the until condition can be a sensor leaf
(sensor distance, grayscale_left, grayscale_center, grayscale_right, battery_voltage, or battery_percent;
op eq, ne, gte, lte, gt, or lt; and numeric value) or nested and/or/xor/not conditions.
Use concrete thresholds for vague words such as near, close, or far.
Example: {"commands":[{"action":"forward","speed":50,"duration_ms":2000},{"action":"stop"}],"reply":"かしこまりました。前進した後に停止します。"}"#;

/// Common response type consumed by the application loop.
#[derive(Debug)]
pub enum LlmResponse {
    Ok {
        request_id: u64,
        commands: Vec<Command>,
        reply: String,
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
}

/// Non-blocking LLM agent shared by every provider.
pub struct ProviderAgent<P> {
    prompt_tx: Sender<Prompt>,
    response_rx: Receiver<LlmResponse>,
    next_request_id: AtomicU64,
    cancelled_through: Arc<AtomicU64>,
    provider: PhantomData<P>,
}

struct Prompt {
    request_id: u64,
    text: String,
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

                let response = match provider
                    .request(&prompt.text)
                    .and_then(|json| parse_commands(&json))
                {
                    Ok((commands, reply)) => LlmResponse::Ok {
                        request_id: prompt.request_id,
                        commands,
                        reply,
                    },
                    Err(message) => LlmResponse::Err {
                        request_id: prompt.request_id,
                        message,
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

    /// Submit a prompt without blocking the caller and return its request ID.
    pub fn submit_prompt(&self, text: String) -> Result<u64, String> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.prompt_tx
            .send(Prompt { request_id, text })
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
                        "duration_ms": { "type": "integer", "minimum": 100, "maximum": 30000 },
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
pub fn parse_commands(response_text: &str) -> Result<(Vec<Command>, String), String> {
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

    for (index, command) in commands.iter_mut().enumerate() {
        normalize_and_validate_command(command, index).map_err(|e| {
            format!(
                "LLMが返したコマンドJSONの安全値検証に失敗: {} (受信内容: {})",
                e, response_text
            )
        })?;
    }

    serde_json::from_value::<CommandListPayload>(value)
        .map(|payload| (payload.commands, payload.reply))
        .map_err(|e| {
            format!(
                "LLMが返したコマンドJSONの解析に失敗: {} (受信内容: {})",
                e, response_text
            )
        })
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

    use super::{LlmProvider, LlmResponse, ProviderAgent, parse_commands};
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
        let commands =
            parse_commands(r#"{"commands":[{"action":"forward","speed":30},{"action":"stop"}]}"#)
                .expect("command JSON should be valid");

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
        let commands = parse_commands(r#"{"commands":[{"action":"forward"}]}"#)
            .expect("safe defaults should be applied");

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
            parse_commands(r#"{"commands":[{"action":"forward","duration_ms":30001}]}"#).is_err()
        );
        assert!(parse_commands(r#"{"commands":[{"action":"forward","speed":101}]}"#).is_err());
        assert!(parse_commands(r#"{"commands":[{"action":"set_cam_pan","angle":91}]}"#).is_err());
    }

    #[test]
    fn parser_defaults_missing_angles_and_normalizes_integral_floats() {
        let commands = parse_commands(
            r#"{"commands":[{"action":"steer","duration_ms":1000.0},{"action":"set_cam_pan"},{"action":"forward","speed":40.0,"duration_ms":500.0}]}"#,
        )
        .expect("safe angle defaults and integral floats should be accepted");

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
        let commands = parse_commands(
            r#"{"commands":[{"action":"forward","speed":20,"duration_ms":500,"until":{"and":[{"sensor":"distance","op":"lt","value":10},{"not":{"sensor":"battery_percent","op":"lt","value":10}}]}}]}"#,
        )
        .expect("nested until conditions should be valid");

        assert!(matches!(commands.first(), Some(Command::Forward { .. })));
    }

    #[test]
    fn parser_rejects_invalid_until_conditions() {
        assert!(
            parse_commands(r#"{"commands":[{"action":"forward","until":{"and":[]}}]}"#).is_err()
        );
        assert!(parse_commands(
            r#"{"commands":[{"action":"forward","until":{"and":[{"sensor":"distance","op":"lt","value":10}],"not":{"sensor":"distance","op":"lt","value":5}}}]}"#
        )
        .is_err());
    }

    #[test]
    fn responses_keep_monotonic_request_ids() {
        let agent = ProviderAgent::with_provider(MockProvider);
        assert_eq!(agent.submit_prompt("ok".to_string()).unwrap(), 1);
        assert_eq!(agent.submit_prompt("error".to_string()).unwrap(), 2);

        let first = receive_response(&agent);
        let second = receive_response(&agent);
        assert!(matches!(first, LlmResponse::Ok { request_id: 1, .. }));
        assert!(matches!(second, LlmResponse::Err { request_id: 2, .. }));
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
