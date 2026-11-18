//! 走行モデル（ONNX）の読み込みと推論。読み込む前にマニフェストで中身を確かめる。
//!
//! モデルのフォルダには、`model.onnx`（と重みの `model.onnx.data`）と、PC の
//! `src.e2e.deploy.bundle` が作る `manifest.json` を置く。マニフェストには、各ファイルの SHA256、
//! 入力の大きさ、正規化の値、意図の順番、出力の名前を書く。ハッシュか入力の大きさが合わなければ
//! 読み込まない（前処理の食い違いや、壊れたファイルで走らないため。ROADMAP の M3
//! 「Pi へのモデルの配置」）。

use super::Intent;
use super::preprocess::Normalization;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;

pub const MANIFEST_FILE: &str = "manifest.json";
pub const MODEL_FILE: &str = "model.onnx";
pub const MANIFEST_FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputSpec {
    pub width: usize,
    pub height: usize,
    /// 画素の値をこれで割ってから正規化する（255）
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// "RGB" だけを扱う
    pub color: String,
    /// "NCHW" だけを扱う
    pub layout: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub format: u32,
    pub model_version: String,
    #[serde(default)]
    pub data_fingerprint: Option<String>,
    #[serde(default)]
    pub model_commit: Option<String>,
    /// ファイル名 → SHA256（16進）
    pub files: BTreeMap<String, String>,
    pub input: InputSpec,
    pub intents: Vec<String>,
    pub outputs: Vec<String>,
    /// 前方が塞がっているかの出力を学習したか。学習していなければ使わない
    #[serde(default)]
    pub blocked_trained: bool,
    /// 舵角の出力（-1〜1）にこれを掛けると度になる
    pub steering_scale_deg: f64,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub software: BTreeMap<String, String>,
}

impl Manifest {
    pub fn normalization(&self) -> Normalization {
        Normalization {
            scale: self.input.scale,
            mean: self.input.mean,
            std: self.input.std,
        }
    }
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("{} を開けません: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|e| format!("{} を読めません: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

pub fn read_manifest(dir: &Path) -> Result<Manifest, String> {
    let path = dir.join(MANIFEST_FILE);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("{} を読めません: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{} の形式が違います: {e}", path.display()))
}

/// マニフェストの中身と、ファイルのハッシュを確かめる。ONNX はまだ読まない
pub fn verify(dir: &Path) -> Result<Manifest, String> {
    let manifest = read_manifest(dir)?;
    if manifest.format != MANIFEST_FORMAT {
        return Err(format!(
            "マニフェストの形式 {} には対応していません（{MANIFEST_FORMAT} のみ）",
            manifest.format
        ));
    }
    if !manifest.files.contains_key(MODEL_FILE) {
        return Err(format!("マニフェストに {MODEL_FILE} がありません"));
    }
    for (name, expected) in &manifest.files {
        if name.contains('/') || name.contains('\\') || name.starts_with('.') {
            return Err(format!("ファイル名が不正です: {name}"));
        }
        let actual = sha256_file(&dir.join(name))?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "{name} のハッシュが合いません（壊れているか、別のファイルです）"
            ));
        }
    }
    let input = &manifest.input;
    if input.color != "RGB" || input.layout != "NCHW" {
        return Err(format!(
            "入力の形式 {} {} には対応していません（RGB NCHW のみ）",
            input.color, input.layout
        ));
    }
    if input.width == 0 || input.height == 0 || input.scale <= 0.0 || input.std.contains(&0.0) {
        return Err("入力の大きさか正規化の値が不正です".to_string());
    }
    let intents: Vec<&str> = manifest.intents.iter().map(String::as_str).collect();
    if intents != Intent::ALL.map(Intent::as_str) {
        return Err(format!("意図の順番が違います: {intents:?}"));
    }
    if manifest.outputs.first().map(String::as_str) != Some("steering") {
        return Err("最初の出力が steering ではありません".to_string());
    }
    if !(1.0..=90.0).contains(&manifest.steering_scale_deg) {
        return Err("steering_scale_deg が不正です".to_string());
    }
    Ok(manifest)
}

/// 推論の結果。`blocked` は学習していないモデルでは None
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Output {
    pub steering: f64,
    pub blocked: Option<f64>,
}

/// 走行モデル。テストでは決まった値を返すものに差し替える
pub trait Steering: Send {
    fn manifest(&self) -> &Manifest;
    fn infer(&mut self, input: &[f32], intent: Intent) -> Result<Output, String>;
}

pub struct TractModel {
    manifest: Manifest,
    plan: TypedRunnableModel<TypedModel>,
    blocked_index: Option<usize>,
}

impl TractModel {
    /// マニフェストとハッシュを確かめ、入力の大きさがモデルと合えば読み込む
    pub fn load(dir: &Path) -> Result<Self, String> {
        let manifest = verify(dir)?;
        let (h, w) = (manifest.input.height, manifest.input.width);
        let model = tract_onnx::onnx()
            .model_for_path(dir.join(MODEL_FILE))
            .map_err(|e| format!("ONNX を読めません: {e}"))?;
        check_input_size(&model, h, w)?;
        let output_count = model.outputs.len();
        if output_count != manifest.outputs.len() {
            return Err(format!(
                "出力の数が違います（モデル {output_count}、マニフェスト {}）",
                manifest.outputs.len()
            ));
        }
        // 入力の大きさは固定しない（固定すると tract 0.21 が最初の畳み込みを解析できなかった。
        // 2026-11-14）。大きさは上の check_input_size と推論の時の形で確かめる
        let plan = model
            .into_optimized()
            .and_then(|m| m.into_runnable())
            .map_err(|e| format!("モデルを準備できません: {e}"))?;
        let blocked_index = if manifest.blocked_trained {
            manifest.outputs.iter().position(|name| name == "blocked")
        } else {
            None
        };
        Ok(Self {
            manifest,
            plan,
            blocked_index,
        })
    }
}

/// モデルの入力の大きさが決まっていれば、マニフェストと同じか確かめる
fn check_input_size(model: &InferenceModel, h: usize, w: usize) -> Result<(), String> {
    let fact = model
        .input_fact(0)
        .map_err(|e| format!("入力の情報を読めません: {e}"))?;
    let Some(shape) = fact.shape.concretize() else {
        return Ok(());
    };
    let dims: Vec<Option<i64>> = shape.iter().map(|d| d.to_i64().ok()).collect();
    if dims.len() != 4 {
        return Err(format!("入力の次元が4ではありません: {dims:?}"));
    }
    if dims[1].is_some_and(|c| c != 3)
        || dims[2].is_some_and(|v| v != h as i64)
        || dims[3].is_some_and(|v| v != w as i64)
    {
        return Err(format!(
            "入力の大きさがマニフェスト（3x{h}x{w}）と違います: {dims:?}"
        ));
    }
    Ok(())
}

impl Steering for TractModel {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn infer(&mut self, input: &[f32], intent: Intent) -> Result<Output, String> {
        let (h, w) = (self.manifest.input.height, self.manifest.input.width);
        let image = tract_ndarray::Array4::from_shape_vec((1, 3, h, w), input.to_vec())
            .map_err(|e| format!("入力の大きさが違います: {e}"))?;
        let intent = tract_ndarray::Array2::from_shape_vec((1, 3), intent.one_hot().to_vec())
            .map_err(|e| e.to_string())?;
        let outputs = self
            .plan
            .run(tvec!(
                Tensor::from(image).into(),
                Tensor::from(intent).into()
            ))
            .map_err(|e| format!("推論に失敗しました: {e}"))?;
        let first = |index: usize| -> Result<f64, String> {
            let view = outputs[index]
                .to_array_view::<f32>()
                .map_err(|e| e.to_string())?;
            view.iter()
                .next()
                .map(|&v| f64::from(v))
                .ok_or_else(|| "出力が空です".to_string())
        };
        let steering = first(0)?;
        let blocked = match self.blocked_index {
            Some(index) => Some(first(index)?),
            None => None,
        };
        Ok(Output { steering, blocked })
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::path::PathBuf;

    /// git にある学習済みのモデル（python/models/v006）から、マニフェスト付きのフォルダを作る
    pub fn v006_bundle(dir: &Path) -> PathBuf {
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../python/models/v006");
        let bundle = dir.join("v006");
        std::fs::create_dir_all(&bundle).unwrap();
        let mut files = BTreeMap::new();
        for name in [MODEL_FILE, "model.onnx.data"] {
            std::fs::copy(source.join(name), bundle.join(name)).unwrap();
            files.insert(name.to_string(), sha256_file(&bundle.join(name)).unwrap());
        }
        let manifest = Manifest {
            format: MANIFEST_FORMAT,
            model_version: "v006".to_string(),
            data_fingerprint: Some("f1eaaabf".to_string()),
            model_commit: None,
            files,
            input: InputSpec {
                width: 160,
                height: 120,
                scale: 255.0,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
                color: "RGB".to_string(),
                layout: "NCHW".to_string(),
            },
            intents: Intent::ALL.map(|i| i.as_str().to_string()).to_vec(),
            outputs: vec!["steering".to_string(), "blocked".to_string()],
            blocked_trained: false,
            steering_scale_deg: 35.0,
            created_at: None,
            software: BTreeMap::new(),
        };
        std::fs::write(
            bundle.join(MANIFEST_FILE),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        bundle
    }

    fn edit_manifest(bundle: &Path, edit: impl FnOnce(&mut Manifest)) {
        let mut manifest = read_manifest(bundle).unwrap();
        edit(&mut manifest);
        std::fs::write(
            bundle.join(MANIFEST_FILE),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn v006_gives_the_same_outputs_as_onnxruntime_on_the_pc() {
        use super::super::preprocess::{decode_jpeg, resize, to_tensor};
        let tmp = tempfile::tempdir().unwrap();
        let mut model = TractModel::load(&v006_bundle(tmp.path())).unwrap();
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/pattern_expected.json"))
                .unwrap();
        let small = resize(
            &decode_jpeg(include_bytes!("../../tests/fixtures/pattern_320x240.jpg")).unwrap(),
            160,
            120,
        );
        let tensor = to_tensor(&small, &model.manifest().normalization());
        for case in expected["v006"].as_array().unwrap() {
            let intent = Intent::parse(case["intent"].as_str().unwrap()).unwrap();
            let out = model.infer(&tensor, intent).unwrap();
            let want = case["steering"].as_f64().unwrap();
            // JPEG の復号の丸めの違いだけ。舵角で 0.5°未満
            assert!(
                (out.steering - want).abs() < 0.015,
                "{intent:?}: {} vs {want}",
                out.steering
            );
            assert_eq!(out.blocked, None); // 学習していない出力は使わない
        }
    }

    #[test]
    fn a_changed_file_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = v006_bundle(tmp.path());
        let mut data = std::fs::read(bundle.join("model.onnx.data")).unwrap();
        data[100] ^= 1;
        std::fs::write(bundle.join("model.onnx.data"), data).unwrap();
        let error = verify(&bundle).unwrap_err();
        assert!(error.contains("ハッシュ"), "{error}");
        assert!(TractModel::load(&bundle).is_err());
    }

    #[test]
    fn a_different_input_size_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = v006_bundle(tmp.path());
        edit_manifest(&bundle, |m| m.input.width = 224);
        let error = TractModel::load(&bundle).err().unwrap();
        assert!(error.contains("入力の大きさ"), "{error}");
    }

    #[test]
    fn manifest_checks() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = v006_bundle(tmp.path());
        assert!(verify(&bundle).is_ok());
        edit_manifest(&bundle, |m| m.intents.swap(1, 2));
        assert!(verify(&bundle).unwrap_err().contains("意図"));
        edit_manifest(&bundle, |m| {
            m.intents.swap(1, 2);
            m.files.insert("../etc".to_string(), "00".to_string());
        });
        assert!(verify(&bundle).unwrap_err().contains("ファイル名"));
        assert!(verify(&tmp.path().join("missing")).is_err());
    }
}
