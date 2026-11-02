# boatrace-lab（verbose-engine）

競艇（ボートレース）で、データ検証を毎日繰り返して**回収率100%超えの戦略を探す実験**のリポジトリ。
運用者（Ban）はスマホのClaudeアプリから会話で指示する。PCは使わない前提。外部DBも使わない。

## 目的と日々のサイクル

- **目的はただ一つ、勝てる方法を見つけること。** レース数・点数に上限はかけない。条件に合うレースがあれば何レースでも出し、無ければ0でよい。1点は当面100円固定。点数（フォーメーション）は検証しながら詰める。前夜に翌日のレースと買い目を決め、朝に人が買い、夜に結果で検証してルールを磨く。
- サイクル（1本のスケジュール実行、毎晩）
  1. 今日の成績を取り込む → `verify.py --write-notes` で的中率・回収率を出す
  2. `notes/daily/` の「気づき」を埋め、効きそうなら `notes/learnings.md` に追記
  3. 翌日の番組表を取り込む → `select_races.py` で買い目を作る → `notes/picks/` に書く
  4. 通知 → Parquet 書き出し → コミット＆プッシュ
- 朝は人が `notes/picks/YYYY-MM-DD.md` を見て買う。買わなかった／変えたときはそのファイルの末尾に書く。

## 約束

- **ファイルに残っているものだけが記憶**。スケジュール実行は毎回まっさら。気づき・決定・ボツ理由は `notes/` に書いてコミットする。
- **評価は回収率が主、的中率は従**。「当たるが儲からない」ルールは採用しない。
- **回収率100%超えは1か月単位で判定する**。暦月（1日〜末日）の払戻合計 ÷ 購入合計 ＝ 月次回収率が 1.0 を超えた月を「勝ち月」と呼ぶ。バックテストは月ごとの回収率・勝ち月数・最悪月を必ず併記する（`verify.py --backtest` が出す）。
- **採用の合格ライン**: 直近12か月で通算回収率 1.0 超え、かつ勝ち月が 8 か月以上（`rules.yaml` の `adopt_*`）。これを満たさないルールは本番の選定に入れない。
- **1日の結果でルールを変えない**。`rules.yaml` の `min_sample_races`（50）未満の試行で良し悪しを断定しない。変更提案は notes に書き、人の判断を待つ。
- **選定ルールは `config/rules.yaml` が唯一の正**。コードにマジックナンバーを埋めない。変更時は version を上げ `notes/decisions.md` に1行。
- **ルール変更は必ずバックテストを添える**：`verify.py --backtest --from --to` で過去3年に当てた回収率を notes に残す。
- **データ取得は速さ優先**。一括取り込みは月単位のプロセスを並列に回してよい（目安は同時12本まで）。同じ日を何度も取り直さない（cache_dir を使う）。エラーが増えたら並列数を下げる。
- **自動で舟券を買わない**。購入は人がやる。このリポジトリは選定と検証まで。

## データの置き方

- 正本は `data/parquet/{table}_{year}.parquet`（git管理、zstd圧縮）
- 作業用の `data/boatrace.db`（SQLite）は gitignore。無ければ `db.py` が Parquet から自動で組み立てる
- データ源は公式ダウンロードサービス（番組表 B / 競走成績 K）。`boatrace-lzh` で取得・パース。2005年〜あり、番組表は翌日分が前夜に出る
- 実行環境のネットワークで `www1.mbrace.or.jp` への到達を許可しておくこと

## ディレクトリ

```
CLAUDE.md               ← このファイル
config/rules.yaml       ← 選定ルール（現行版）
scripts/
  common.py             ← パス・日付・場コード
  db.py                 ← SQLite⇄Parquet、stats、その場SQL
  ingest.py             ← 番組表/成績の取り込み（単日・期間）
  ingest_fan.py         ← 選手期別成績（fanYYMM.lzh、2001年〜全期）
  scrape_web.py         ← boatrace.jp の締切時オッズ・直前情報
  select_races.py       ← 翌日の買い目を作る
  verify.py             ← 検証、移動平均、バックテスト
  notify.py             ← Slack 通知（未設定なら標準出力）
data/parquet/           ← データ正本
notes/
  learnings.md          ← 気づき（追記のみ）
  decisions.md          ← ルール変更・設計判断
  daily/YYYY-MM-DD.md   ← 日次の検証
  picks/YYYY-MM-DD.md   ← 翌日の買い目（朝に人が見る）
.claude/skills/
  nightly/              ← 毎晩の全手順
  backfill/             ← 過去データの一括取り込み
```

## テーブル（SQLite / Parquet 共通）

- `entries`（出走表）hd, jcd, rno, lane, racer_no, racer_name, grade, motor_no, boat_no, close_time, race_name, data(JSON)
- `results`（着順）hd, jcd, rno, lane, position, data(JSON)
- `payouts`（払戻）hd, jcd, rno, ticket_type, combination, payout（100円あたり）
- `picks`（買い目）hd, jcd, rno, ticket_type, combination, amount, rules_version, reason, paper
- `verifications`（日次）hd, rules_version, races, bets, hits, bet_total, payout_total, hit_rate, roi, note
- `odds`（締切時オッズ）hd, jcd, rno, ticket_type, combination, odds, odds_max（複勝の上限）
- `beforeinfo`（直前情報）hd, jcd, rno, lane, exhibition_time, tilt, start_course, start_st, data(JSON: 体重・部品交換・水面気象など)
- `racer_stats`（期別成績）hd(=算出期間の最終日), racer_no, year, term, grade, win_rate, top2_rate, avg_st, data(JSON: コース別の進入・着順回数・平均ST・c1_win_rate など)

### いつ手に入る情報か（リークに注意）

| データ | 手に入る時点 | 前夜の選定に使えるか |
|---|---|---|
| entries（番組表） | 前日夜 | 使える |
| racer_stats（期別成績） | 算出期間の最終日（hd）より後 | レース日より前の hd の最新行だけ使う |
| results / payouts の過去分 | 各レース後 | 過去日の分だけ使う（モーター・選手の直近成績など） |
| results の展示タイム・進入・ST | レース当日 | 使えない（当日運用にするなら beforeinfo を使う） |
| beforeinfo（直前情報） | レース当日・展示後 | 使えない |
| odds（締切時オッズ） | 締切時 | 使えない。市場の歪み（どこが過小評価か）を調べる用 |

`data` 列には元レコードの全項目を JSON で残している。項目名が確定していない列（1コース1着率、モーター2連率など）は、ここから取り出して使う。

## 用語

- jcd：場コード（01桐生〜24大村、`common.VENUES`）
- rno：レース番号（1〜12）、hd：開催日（YYYYMMDD）、lane：艇番（1〜6）
- 回収率：払戻合計 ÷ 購入額合計（1.0 が損益分岐。控除率約25%なので何もしなければ0.75に収束する）

## まだ決まっていないこと

- boatrace-lzh の各レコードの項目名（初回取り込みで `ingest.py` の `_pick()` 候補名を実データに合わせる）
- 通知先（Slack Webhook を環境変数 SLACK_WEBHOOK_URL に入れる or 当面は notes だけ）
- 実行環境での boatrace-lzh のインストール可否（LZH 解凍に p7zip-full か lha が要る）
