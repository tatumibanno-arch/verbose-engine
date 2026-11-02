---
name: morning
description: 朝の定型処理。出走表を取得し、rules.yaml で今日の対象レースを選定して通知する。「朝のやつ」「今日の選定」でも呼ぶ。
---

# 朝の手順

1. `pip install -r requirements.txt`（初回のみ。入っていれば飛ばす）
2. `python scripts/fetch_racelist.py` を実行。skip が大量に出るのは非開催場なので正常。1件も ok が出なければネットワーク許可か公式サイトの構造変更を疑う。
3. `python scripts/select_races.py` を実行。
4. `python scripts/notify.py --mode morning` を実行。
5. 失敗した工程があれば `notes/daily/YYYY-MM-DD.md` の冒頭に「朝: ○○で失敗（原因）」を書いてコミットする。成功時はコミット不要。

## やらないこと

- rules.yaml をこの手順の中で書き換えない（ルール変更は人の指示があるときだけ）
- 公式サイトへの再取得を3回以上繰り返さない
