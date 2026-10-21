# 第3章：GeminiのStructured Outputsがもたらす「型安全」な解決


## 救世主：Structured Outputs（構造化出力）とは？

前章で紹介した「JSONが崩れてロボットが誤動作する問題」を根本から解決するのが、Gemini APIの **「Structured Outputs（構造化出力）」** という機能です。

- **公式ドキュメント**: [Structured Outputs | Gemini API | Google AI for Developers](https://ai.google.dev/gemini-api/docs/structured-outputs)

これは、プロンプトで「JSONで出力して」とお願いするのではなく、**「出力するJSONの構造（スキーマ）をあらかじめ定義してAPIに送り、モデルにそのスキーマを100%厳守させる」** 機能です。

例えば、以下のように「キー名」や「データの型」「必須項目」を明示したスキーマ（JSON Schema）をリクエストに含めて送信します。

```json
{
  "type": "OBJECT",
  "properties": {
    "commands": {
      "type": "ARRAY",
      "items": {
        "type": "OBJECT",
        "properties": {
          "action": {
            "type": "STRING",
            "enum": ["forward", "backward", "stop"]
          },
          "speed": {
            "type": "INTEGER"
          }
        },
        "required": ["action"]
      }
    }
  },
  "required": ["commands"]
}
```

#### ポイント解説
- **`properties` と `type`**: オブジェクトに含まれるキー名（`commands`, `action`, `speed`）とそのデータ型を定義します。
- **`enum`**: 文字列に入る値を `forward`, `backward`, `stop` のいずれかに限定します。
- **`required`**: そのキーが必ずレスポンスに含まれることを強制します。

このスキーマをAPI側に渡すことで、Geminiは**レスポンスを生成する際にスキーマ外のキーを出力したり、挨拶文を混ぜたりすることが構造上できなくなります。** これにより、返ってくるデータは常に定義した通りのクリーンなJSONオブジェクトになることが保証されます。

---

## Rustの「型システム」との相乗効果

この構造化出力は、**Rust言語と非常に強力なシナジー（相乗効果）を発揮します。**

Rustには、JSONデータをRustの構造体や列挙型（`enum`）に一発で変換（デシリアライズ）できる非常に強力なライブラリ **`serde`** があります。

通常、LLMの出力が不安定だと、`serde` は「型やキーが合わない」としてすぐにパースエラー（エラー返却）を起こしてしまいます。しかし、Geminiの構造化出力によって「スキーマ通りであること」がAPIレベルで100%保証されると、以下のようなメリットが生まれます。

1. **パースエラーによる強制終了の撲滅**
   - 届くデータが必ずスキーマに適合しているため、Rust側での `serde_json::from_str` が失敗することがほぼゼロになります。
2. **無駄な「例外処理コード」の削減**
   - 挨拶のトリム処理、マークダウン記法の除去、キーの揺らぎに対する分岐処理など、これまでパースエラー対策のために書いていた泥臭いコードがすべて不要になり、ビジネスロジックに集中できます。
3. **Rustの型安全性の恩恵をフルに受ける**
   - LLMから取得したコマンド情報を、直接Rustの型安全な `Command` enum にマッピングしてそのままロボットのモーター駆動関数などに流し込めるようになります。
