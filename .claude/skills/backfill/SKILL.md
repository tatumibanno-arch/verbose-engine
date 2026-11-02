---
name: backfill
description: 過去データ（番組表・成績）を期間指定で一括取り込みし、Parquetに書き出してコミットする。「3年分入れて」「過去データ取って」でも呼ぶ。
---

# 過去データの一括取り込み

1. `pip install -r requirements.txt`。`lha` も `7z` も無ければ `apt-get install -y p7zip-full`。
2. **まず1日だけ試す**：`python scripts/ingest.py performance --date <直近の開催日>` と `schedule`。
   - 0件なら `boatrace-lzh` の項目名が想定と違う。`python -c "from boatrace_lzh import *; ..."` で1レコードの属性を表示し、`scripts/ingest.py` の `_pick()` 候補名を直す。
   - 直したら `notes/decisions.md` に「ingest の項目名を確定（実データ基準）」と1行。
3. **期間を分けて取り込む**：1か月ずつ `--from/--to` で回す。1本のコマンドで3年分を回さない（途中で落ちたときに再開しやすいように）。
   - 例：`python scripts/ingest.py performance --from 20231101 --to 20231130`
   - `schedule` も同じ期間で。
4. **件数確認**：`python scripts/db.py stats`。1開催日あたり entries ≈ 開催場数×12×6 を目安に抜けを見る。
5. **書き出しとコミット**：`python scripts/db.py export` → `git add data/parquet && git commit -m "backfill: <期間>" && git push`
   - 1ファイル100MBを超えそうなら止めて報告する（年で分けているので通常は超えない）。
6. 終わったら `python scripts/verify.py --backtest --from <開始> --to <終了>` を回し、現行ルールの過去回収率を `notes/learnings.md` に記録する。これが以後のルール改善の基準線になる。
