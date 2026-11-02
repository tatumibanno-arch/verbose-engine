# boatrace-lab（verbose-engine）

競艇（ボートレース）の出走表・結果データを毎日自動で集め、レース選定ロジックを検証し続けるためのリポジトリ。
運用者はスマホのClaudeアプリから会話で指示する。PCは使わない前提で設計する。

## このリポジトリの約束

- **ファイルに残っているものだけが記憶**。スケジュール実行は毎回まっさらなセッションで始まる。気づき・決定・ボツ理由は必ず `notes/` に書いてコミットする。
- **検証は回収率ベース**。的中率だけで判断しない。「当たるが儲からない」ロジックを採用しない。
- **選定ルールは `config/rules.yaml` が唯一の正**。コードにマジックナンバーを埋めない。ルール変更は yaml を書き換え、`notes/learnings.md` に「いつ・なぜ変えたか」を1行残す。
- **データ取得は礼儀正しく**。公式サイト（boatrace.jp）へのアクセスは1ページごとに2秒以上の間隔、1日の取得回数は朝1回・夜1回を原則とする。User-Agent を明示し、失敗時のリトライは最大2回。
- **破壊的変更は保守的に**。Supabase のテーブル削除・大量更新はしない。スキーマ変更は `supabase/migrations/` に残す。

## ディレクトリ

```
CLAUDE.md               ← このファイル。毎セッション最初に読まれる
config/rules.yaml       ← レース選定ルール（現行版）
scripts/
  fetch_racelist.py     ← 出走表の取得 → Supabase
  fetch_results.py      ← 結果・払戻の取得 → Supabase
  select_races.py       ← rules.yaml に従って今日の対象レースを選定
  verify.py             ← 前日の選定を結果と突き合わせ、的中率・回収率を算出
  notify.py             ← 選定結果・検証結果を通知先へ送る
notes/
  learnings.md          ← 検証から得た気づき（追記のみ、日付つき）
  decisions.md          ← ルール変更・設計判断の記録
  daily/YYYY-MM-DD.md   ← 日次の検証ログ
.claude/skills/
  morning/              ← 朝の手順（取得→選定→通知）
  evening/              ← 夜の手順（結果取得→検証→学び追記→コミット）
supabase/migrations/    ← スキーマ
```

## 日次の流れ

- **朝（出走表公開後）**：`fetch_racelist.py` → `select_races.py` → `notify.py --mode morning`
- **夜（全レース終了後）**：`fetch_results.py` → `verify.py` → `notes/daily/` と `notes/learnings.md` を更新 → `notify.py --mode evening` → コミット＆プッシュ

## 用語

- jcd：場コード（01桐生〜24大村）
- rno：レース番号（1〜12）
- hd：開催日（YYYYMMDD）
- 回収率：払戻合計 ÷ 購入額合計（100%が損益分岐）

## まだ決まっていないこと（会話で決める）

- Supabase に蓄積する項目の確定（出走表のどこまで取るか）
- 選定ルールの初版（`config/rules.yaml` は雛形のまま）
- 通知先（Slack / Notion）
- 実行環境のネットワーク許可（boatrace.jp への到達を環境設定で許可する必要あり）
