//! `picar-x-rust model ...`: Pi の走行モデルの版の管理（走行サーバーとは別に実行する）。
//!
//! ```text
//! picar-x-rust model list                    版の一覧（* が使用中）
//! picar-x-rust model install <dir> [--activate]  PC から送ったフォルダを確かめて版として加える
//! picar-x-rust model activate <版>           版を使用中にする
//! picar-x-rust model rollback                1つ前に使っていた版へ戻す
//! picar-x-rust model verify                  使用中の版のハッシュと中身を確かめる
//! ```
//!
//! モデルのフォルダは `PICARX_MODELS_DIR`（既定は `~/picarx/models`）。切り替えは次の自律走行の
//! 開始から使われる。

use crate::autopilot::model::verify;
use crate::autopilot::models_dir_from_env;
use crate::autopilot::store::ModelStore;
use std::path::Path;

const USAGE: &str = "使い方: picar-x-rust model <list | install <dir> [--activate] | activate <version> | rollback | verify>";

pub fn run(args: &[String]) -> i32 {
    let store = ModelStore::new(models_dir_from_env());
    let result = match args.first().map(String::as_str) {
        Some("list") => {
            let versions = store.list();
            if versions.is_empty() {
                println!("版がありません（{}）", store.root().display());
            }
            for v in versions {
                println!("{} {}", if v.current { "*" } else { " " }, v.version);
            }
            Ok(())
        }
        Some("install") => match args.get(1) {
            Some(dir) => store.install(Path::new(dir)).and_then(|version| {
                println!("✓ 版 {version} を加えました");
                if args.iter().any(|a| a == "--activate") {
                    store.activate(&version)?;
                    println!("✓ 版 {version} を使用中にしました（次の自律走行から使います）");
                }
                Ok(())
            }),
            None => Err(USAGE.to_string()),
        },
        Some("activate") => match args.get(1) {
            Some(version) => store.activate(version).map(|()| {
                println!("✓ 版 {version} を使用中にしました（次の自律走行から使います）");
            }),
            None => Err(USAGE.to_string()),
        },
        Some("rollback") => store.rollback().map(|version| {
            println!("✓ 版 {version} へ戻しました（次の自律走行から使います）");
        }),
        Some("verify") => match store.current() {
            Some((version, dir)) => verify(&dir).map(|m| {
                println!(
                    "✓ 版 {version} は正しい中身です（入力 {}x{}、塞がっているかの出力: {}）",
                    m.input.width,
                    m.input.height,
                    if m.blocked_trained {
                        "使う"
                    } else {
                        "使わない"
                    }
                );
            }),
            None => Err("使用中の版がありません".to_string()),
        },
        _ => Err(USAGE.to_string()),
    };
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("✗ {error}");
            1
        }
    }
}
