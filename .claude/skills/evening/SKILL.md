---
name: evening
description: 夜の定型処理。結果と払戻を取得し、朝の選定を検証して的中率・回収率を出し、気づきを notes に残してコミットする。「夜のやつ」「検証して」でも呼ぶ。
---

# 夜の手順

1. `pip install -r requirements.txt`（初回のみ）
2. `python scripts/fetch_results.py` を実行。
3. `python scripts/verify.py --write-notes` を実行。`notes/daily/YYYY-MM-DD.md` が生成される。
4. 生成された日次ノートの「気づき」欄を埋める。書くのは事実と仮説だけ：
   - 回収率が100%を割った/超えた主因はどのレースか
   - 選定ルールが拾い損ねた・拾いすぎたパターンはあるか
   - 直近7日の回収率の推移（verifications テーブルから）
5. 明日以降も効きそうな発見があれば `notes/learnings.md` に日付つきで追記する。1日の結果だけで断定しない（「傾向あり、要継続観察」の書き方でよい）。
6. `python scripts/notify.py --mode evening` を実行。
7. `git add notes && git commit -m "verify: YYYY-MM-DD" && git push`

## やらないこと

- rules.yaml を自動で書き換えない。変更提案は notes に書き、人の判断を待つ
- 1日の結果で「このルールは駄目」と結論づけない
