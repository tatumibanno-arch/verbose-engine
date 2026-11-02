---
name: nightly
description: 毎晩の全手順。今日の成績で検証→気づきを記録→翌日の買い目を作成→通知→Parquet書き出し→コミット。「夜のやつ」「検証して明日の選定して」でも呼ぶ。
---

# 毎晩の手順（この順番で、止まらずに最後まで）

0. `pip install -r requirements.txt`（入っていれば一瞬で終わる）。`lha` も `7z` も無ければ `apt-get install -y p7zip-full` を試す。
1. **今日の成績を取り込む**：`python scripts/ingest.py performance --date <今日YYYYMMDD>`
   - ERROR が出て0件なら、まだ全レース終わっていない可能性。`notes/daily/<今日>.md` に「成績未取得」とだけ書いて 4 へ進む。
2. **検証**：`python scripts/verify.py --date <今日> --write-notes`
   - `notes/daily/<今日>.md` が生成される（ルール別に回収率・今月累計・直近7日/30日）。
3. **気づきを書く**（`notes/daily/<今日>.md` の「気づき」欄）。事実と仮説のみ：
   - 回収率を動かした主因のレース（高配当の的中 or 本命の取りこぼし）
   - ルールが拾い損ねた／拾いすぎたパターン
   - 直近7日・30日の回収率の向き（ノートに自動で出ている）
   - `notes/picks/<今日>.md` の「実購入メモ」に人の記入があれば、紙の上の結果と実購入の差を1行
   - 翌日以降も効きそうなら `notes/learnings.md` に日付つきで追記。断定しない（「傾向あり、要継続」でよい）。
4. **翌日の番組表を取り込む**：`python scripts/ingest.py schedule --date <明日YYYYMMDD>`
5. **モデルの用意（月初だけ時間がかかる）**：`python scripts/model.py ensure --date <明日>`
   - `data/models/tansho_ev/<明日の年月>/` が無ければ、その月のモデルを学習して保存する（直前24か月の特徴量づくり＋学習。目安 10〜15分）。あれば一瞬で終わる。
   - 失敗しても止まらずに 6 へ進む（単勝ルールは「モデル未学習」と picks ファイルに出て 0 点になる）。失敗は `notes/daily/<今日>.md` に1行。
6. **買い目を作る**：`python scripts/select_races.py --date <明日>`
   - ルールごと（`lane1_a1`、`strategies:` の `tansho_ev_v1` など）に買い目を作り、`notes/picks/<明日>.md` にルール別の見出しで出る。候補0件のルールはその旨が出る。
   - 単勝ルールは翌日分の1着確率・予想配当を `data/models/tansho_ev/history.parquet` に追記する（翌月の学習に使う。消さない）。目安 1〜2分。
7. **通知**：`python scripts/notify.py --mode verify` と `--mode picks`
8. **Parquet 書き出し**：`python scripts/db.py export`
9. **コミット＆プッシュ**：`git add data/parquet data/models notes && git commit -m "nightly: <今日>" && git push`

月の集計（ルール別の月次回収率）は `python scripts/verify.py --monthly`。日次ノートもルール別に出る。

## やらないこと

- `config/rules.yaml` を自動で書き換えない。提案は `notes/decisions.md` の末尾に「提案:」として書き、人の判断を待つ
- 1日の結果でルールを評価しない（`min_sample_races` 未満は様子見）
- 同じ日の成績を3回以上取り直さない
