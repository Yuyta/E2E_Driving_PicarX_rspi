//! Pi に置く走行モデルの版の管理（ROADMAP の M3「Pi へのモデルの配置」）。
//!
//! ```text
//! <root>/versions/<版>/   manifest.json, model.onnx, model.onnx.data
//! <root>/current          → versions/<使用中の版>（シンボリックリンク）
//! <root>/history.json     使用中にした版の順番（新しいものが最後）
//! ```
//!
//! 版の追加、切り替え、戻しは、走行サーバーとは別に `picar-x-rust model ...` で行う。走行サーバーは
//! `current` を読むだけなので、モデルのフォルダへの書き込み権限は要らない。切り替えはリンクの
//! 張り替え（rename）で一度に行い、使用中の版と、その前の2つの版を残す。

use super::model::{MANIFEST_FILE, verify};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const CURRENT_LINK: &str = "current";
pub const VERSIONS_DIR: &str = "versions";
pub const HISTORY_FILE: &str = "history.json";
/// 使用中の版のほかに残す、前の版の数
pub const KEEP_PREVIOUS: usize = 2;

#[derive(Debug, Default, Serialize, Deserialize)]
struct History {
    /// 使用中にした版。新しいものが最後
    activated: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VersionInfo {
    pub version: String,
    pub current: bool,
}

pub struct ModelStore {
    root: PathBuf,
}

fn valid_version(version: &str) -> Result<(), String> {
    let ok = !version.is_empty()
        && version.len() <= 64
        && !version.starts_with('.')
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    if ok {
        Ok(())
    } else {
        Err(format!("版の名前が不正です: {version:?}"))
    }
}

fn io_error(action: &str, path: &Path, error: std::io::Error) -> String {
    format!("{action} に失敗しました（{}）: {error}", path.display())
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    fs::create_dir_all(to).map_err(|e| io_error("フォルダの作成", to, e))?;
    for entry in fs::read_dir(from).map_err(|e| io_error("フォルダの読み取り", from, e))? {
        let entry = entry.map_err(|e| io_error("フォルダの読み取り", from, e))?;
        let path = entry.path();
        if path.is_file() {
            let target = to.join(entry.file_name());
            fs::copy(&path, &target).map_err(|e| io_error("コピー", &path, e))?;
        }
    }
    Ok(())
}

impl ModelStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn versions_dir(&self) -> PathBuf {
        self.root.join(VERSIONS_DIR)
    }

    pub fn version_dir(&self, version: &str) -> PathBuf {
        self.versions_dir().join(version)
    }

    /// 使用中の版の名前とフォルダ
    pub fn current(&self) -> Option<(String, PathBuf)> {
        let target = fs::read_link(self.root.join(CURRENT_LINK)).ok()?;
        let version = target.file_name()?.to_str()?.to_string();
        let dir = self.version_dir(&version);
        dir.is_dir().then_some((version, dir))
    }

    fn history(&self) -> History {
        fs::read_to_string(self.root.join(HISTORY_FILE))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    fn save_history(&self, history: &History) -> Result<(), String> {
        let path = self.root.join(HISTORY_FILE);
        let tmp = self.root.join(format!("{HISTORY_FILE}.tmp"));
        let text = serde_json::to_string_pretty(history).map_err(|e| e.to_string())?;
        fs::write(&tmp, text).map_err(|e| io_error("書き込み", &tmp, e))?;
        fs::rename(&tmp, &path).map_err(|e| io_error("書き込み", &path, e))
    }

    pub fn list(&self) -> Vec<VersionInfo> {
        let current = self.current().map(|(v, _)| v);
        let mut versions: Vec<VersionInfo> = fs::read_dir(self.versions_dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join(MANIFEST_FILE).is_file())
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .map(|version| VersionInfo {
                current: current.as_deref() == Some(version.as_str()),
                version,
            })
            .collect();
        versions.sort_by(|a, b| a.version.cmp(&b.version));
        versions
    }

    /// PC から送ったフォルダを確かめてから版として加える。同じ版が同じ中身ですでにあれば何もしない
    pub fn install(&self, bundle: &Path) -> Result<String, String> {
        let manifest = verify(bundle)?;
        let version = manifest.model_version.clone();
        valid_version(&version)?;
        let target = self.version_dir(&version);
        if target.exists() {
            return match verify(&target) {
                Ok(existing) if existing.files == manifest.files => Ok(version),
                _ => Err(format!(
                    "版 {version} はすでに別の中身で入っています。版の名前を変えてください"
                )),
            };
        }
        let staging = self.versions_dir().join(format!(".{version}.incoming"));
        let _ = fs::remove_dir_all(&staging);
        copy_dir(bundle, &staging)?;
        // コピーした後にもう一度確かめる（転送やディスクの誤りを使用前に見つける）
        if let Err(error) = verify(&staging) {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        fs::rename(&staging, &target).map_err(|e| io_error("版の追加", &target, e))?;
        Ok(version)
    }

    /// 版を使用中にする。リンクを張り替えて一度に切り替え、古い版を片付ける
    pub fn activate(&self, version: &str) -> Result<(), String> {
        valid_version(version)?;
        let dir = self.version_dir(version);
        verify(&dir)?;
        let link = self.root.join(CURRENT_LINK);
        let tmp = self.root.join(format!("{CURRENT_LINK}.tmp"));
        let _ = fs::remove_file(&tmp);
        std::os::unix::fs::symlink(Path::new(VERSIONS_DIR).join(version), &tmp)
            .map_err(|e| io_error("リンクの作成", &tmp, e))?;
        fs::rename(&tmp, &link).map_err(|e| io_error("リンクの張り替え", &link, e))?;

        let mut history = self.history();
        if history.activated.last().map(String::as_str) != Some(version) {
            history.activated.push(version.to_string());
        }
        self.save_history(&history)?;
        self.prune(&history);
        Ok(())
    }

    /// 1つ前に使っていた版へ戻す。戻した版を返す
    pub fn rollback(&self) -> Result<String, String> {
        let history = self.history();
        let current = self.current().map(|(v, _)| v);
        let previous = history
            .activated
            .iter()
            .rev()
            .filter(|v| Some(v.as_str()) != current.as_deref())
            .find(|v| self.version_dir(v).join(MANIFEST_FILE).is_file())
            .cloned()
            .ok_or("戻せる前の版がありません")?;
        self.activate(&previous)?;
        Ok(previous)
    }

    /// 使用中の版と、その前の KEEP_PREVIOUS 個の版だけを残す
    fn prune(&self, history: &History) {
        let mut keep: Vec<&str> = Vec::new();
        for version in history.activated.iter().rev() {
            if !keep.contains(&version.as_str()) {
                keep.push(version);
            }
            if keep.len() > KEEP_PREVIOUS {
                break;
            }
        }
        for info in self.list() {
            if !keep.contains(&info.version.as_str()) && !info.current {
                let _ = fs::remove_dir_all(self.version_dir(&info.version));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::tests::v006_bundle;
    use super::super::model::{Manifest, read_manifest};
    use super::*;

    /// v006 の中身で、版の名前だけ変えたフォルダ
    fn bundle_named(tmp: &Path, base: &Path, version: &str) -> PathBuf {
        let dir = tmp.join(format!("bundle_{version}"));
        copy_dir(base, &dir).unwrap();
        let mut manifest: Manifest = read_manifest(&dir).unwrap();
        manifest.model_version = version.to_string();
        fs::write(
            dir.join(MANIFEST_FILE),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn install_activate_and_roll_back() {
        let tmp = tempfile::tempdir().unwrap();
        let base = v006_bundle(tmp.path());
        let store = ModelStore::new(tmp.path().join("models"));
        assert!(store.current().is_none());

        for version in ["v006", "v007"] {
            let installed = store
                .install(&bundle_named(tmp.path(), &base, version))
                .unwrap();
            store.activate(&installed).unwrap();
        }
        assert_eq!(store.current().unwrap().0, "v007");
        assert_eq!(store.rollback().unwrap(), "v006");
        assert_eq!(store.current().unwrap().0, "v006");
        // もう一度戻すと v007 へ（1つ前に使っていた版）
        assert_eq!(store.rollback().unwrap(), "v007");
    }

    #[test]
    fn keeps_the_current_and_two_previous_versions() {
        let tmp = tempfile::tempdir().unwrap();
        let base = v006_bundle(tmp.path());
        let store = ModelStore::new(tmp.path().join("models"));
        for version in ["v1", "v2", "v3", "v4"] {
            let installed = store
                .install(&bundle_named(tmp.path(), &base, version))
                .unwrap();
            store.activate(&installed).unwrap();
        }
        let versions: Vec<String> = store.list().into_iter().map(|v| v.version).collect();
        assert_eq!(versions, vec!["v2", "v3", "v4"]);
        assert!(store.list().iter().any(|v| v.current && v.version == "v4"));
    }

    #[test]
    fn a_broken_bundle_is_not_installed_and_names_are_checked() {
        let tmp = tempfile::tempdir().unwrap();
        let base = v006_bundle(tmp.path());
        let store = ModelStore::new(tmp.path().join("models"));
        let broken = bundle_named(tmp.path(), &base, "v9");
        fs::write(broken.join("model.onnx.data"), b"broken").unwrap();
        assert!(store.install(&broken).is_err());
        assert!(store.list().is_empty());

        let bad_name = bundle_named(tmp.path(), &base, "../evil");
        assert!(store.install(&bad_name).unwrap_err().contains("版の名前"));
        assert!(store.activate("missing").is_err());
        assert!(store.rollback().is_err());
    }

    #[test]
    fn reinstalling_the_same_version_is_allowed_only_with_the_same_files() {
        let tmp = tempfile::tempdir().unwrap();
        let base = v006_bundle(tmp.path());
        let store = ModelStore::new(tmp.path().join("models"));
        assert_eq!(store.install(&base).unwrap(), "v006");
        assert_eq!(store.install(&base).unwrap(), "v006");
        let other = bundle_named(tmp.path(), &base, "v006");
        let mut manifest = read_manifest(&other).unwrap();
        manifest
            .files
            .insert("model.onnx".to_string(), "00".to_string());
        fs::write(
            other.join(MANIFEST_FILE),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        assert!(store.install(&other).is_err());
    }
}
