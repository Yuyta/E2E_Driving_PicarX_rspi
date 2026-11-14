//! 1回の自律走行の記録。PC の `src/e2e/intervention.py` の RunLog と同じ形式で書くので、
//! PC へ持ってくれば `src.e2e.evaluation.interventions` でモデル別の介入率を集計できる。
//!
//! `<runs>/<run_id>/run.json` は介入のたびに書き直し、`interventions.jsonl` に1行ずつ足す。
//! 人の介入（引き継ぎ、停止の指示、緊急停止）と、安全のための自動停止を分ける。

use super::model::Manifest;
use serde_json::{Value, json};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const HUMAN_TRIGGERS: [&str; 2] = ["human_takeover", "manual_intervention"];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Counts {
    pub human: u32,
    pub safety: u32,
}

pub struct RunLog {
    pub run_id: String,
    dir: PathBuf,
    info: Value,
    started_at: f64,
    pub autonomous_s: f64,
    pub human_s: f64,
    pub counts: Counts,
}

pub fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

impl RunLog {
    pub fn create(
        base: &Path,
        manifest: &Manifest,
        settings: Value,
        now: f64,
    ) -> Result<Self, String> {
        let local = chrono::DateTime::from_timestamp(now as i64, 0)
            .map(|t| t.with_timezone(&chrono::Local))
            .ok_or("時刻が不正です")?;
        let mut run_id = local.format("%Y%m%d_%H%M%S").to_string();
        let mut dir = base.join(&run_id);
        let mut n = 1;
        while dir.exists() {
            n += 1;
            run_id = format!("{}_{n}", local.format("%Y%m%d_%H%M%S"));
            dir = base.join(&run_id);
        }
        fs::create_dir_all(&dir).map_err(|e| format!("{} を作れません: {e}", dir.display()))?;
        let info = json!({
            "run_id": run_id,
            "model_path": format!("pi:{}", manifest.model_version),
            "model_version": manifest.model_version,
            "data_fingerprint": manifest.data_fingerprint,
            "model_commit": manifest.model_commit,
            "code": {"server": format!("picar-x-rust {}", env!("CARGO_PKG_VERSION"))},
            "where": "pi",
            "settings": settings,
        });
        let log = Self {
            run_id,
            dir,
            info,
            started_at: now,
            autonomous_s: 0.0,
            human_s: 0.0,
            counts: Counts::default(),
        };
        log.write(now, false)?;
        Ok(log)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn add_time(&mut self, seconds: f64, human: bool) {
        let s = seconds.max(0.0);
        if human {
            self.human_s += s;
        } else {
            self.autonomous_s += s;
        }
    }

    pub fn intervention(
        &mut self,
        trigger: &str,
        now: f64,
        note: &str,
        extra: Value,
    ) -> Result<(), String> {
        let human = HUMAN_TRIGGERS.contains(&trigger);
        if human {
            self.counts.human += 1;
        } else {
            self.counts.safety += 1;
        }
        let mut record = json!({
            "time": now,
            "elapsed_s": round3(now - self.started_at),
            "autonomous_s": round3(self.autonomous_s),
            "trigger": trigger,
            "kind": if human { "human" } else { "safety" },
            "note": note,
        });
        if let (Some(record), Value::Object(extra)) = (record.as_object_mut(), extra) {
            record.extend(extra);
        }
        let path = self.dir.join("interventions.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{} を開けません: {e}", path.display()))?;
        writeln!(file, "{record}").map_err(|e| e.to_string())?;
        self.write(now, false)
    }

    pub fn close(&self, now: f64) -> Result<(), String> {
        self.write(now, true)
    }

    fn write(&self, now: f64, ended: bool) -> Result<(), String> {
        let mut data = self.info.clone();
        if let Some(obj) = data.as_object_mut() {
            obj.insert("started_at".into(), json!(self.started_at));
            obj.insert("updated_at".into(), json!(now));
            obj.insert("ended".into(), json!(ended));
            obj.insert("autonomous_s".into(), json!(round3(self.autonomous_s)));
            obj.insert("human_s".into(), json!(round3(self.human_s)));
            obj.insert("interventions".into(), json!(self.counts));
        }
        let path = self.dir.join("run.json");
        let tmp = self.dir.join("run.json.tmp");
        let text = serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?;
        fs::write(&tmp, text + "\n").map_err(|e| format!("{} に書けません: {e}", tmp.display()))?;
        fs::rename(&tmp, &path).map_err(|e| format!("{} に書けません: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::read_manifest;
    use super::super::model::tests::v006_bundle;
    use super::*;

    #[test]
    fn writes_the_same_fields_as_the_pc_run_log() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = read_manifest(&v006_bundle(tmp.path())).unwrap();
        let mut log = RunLog::create(
            &tmp.path().join("runs"),
            &manifest,
            json!({"speed": 50}),
            1000.0,
        )
        .unwrap();
        log.add_time(30.0, false);
        log.add_time(5.0, true);
        log.intervention("obstacle_stop", 1031.0, "obstacle", json!({}))
            .unwrap();
        log.intervention(
            "human_takeover",
            1032.0,
            "manual command",
            json!({"takeover_id": 1}),
        )
        .unwrap();
        log.close(1040.0).unwrap();

        let run: Value =
            serde_json::from_str(&fs::read_to_string(log.dir().join("run.json")).unwrap()).unwrap();
        assert_eq!(run["model_version"], "v006");
        assert_eq!(run["data_fingerprint"], "f1eaaabf");
        assert_eq!(run["interventions"], json!({"human": 1, "safety": 1}));
        assert_eq!(run["autonomous_s"], 30.0);
        assert_eq!(run["human_s"], 5.0);
        assert_eq!(run["ended"], true);
        assert_eq!(run["settings"]["speed"], 50);
        let lines: Vec<Value> = fs::read_to_string(log.dir().join("interventions.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines[0]["kind"], "safety");
        assert_eq!(lines[1]["kind"], "human");
        assert_eq!(lines[1]["takeover_id"], 1);

        // 同じ秒に2回始めても別のフォルダにする
        let second =
            RunLog::create(&tmp.path().join("runs"), &manifest, json!({}), 1000.0).unwrap();
        assert_ne!(second.run_id, log.run_id);
    }
}
