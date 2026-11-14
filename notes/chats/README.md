# チャットの分け方（2026-11-14、運用者の希望）
運用者がこんがらがらないよう、話題ごとにチャットを分ける。どのチャットも同じリポジトリ・同じブランチ（claude/tender-newton-ymc8j0）に書く。作業の前に必ず `git pull --rebase --autostash origin claude/tender-newton-ymc8j0`、終わったらコミットして `git push origin HEAD:claude/tender-newton-ymc8j0`。

| チャット | 扱うこと | 主なファイル |
|---|---|---|
| 転がし（4段）※元のチャット | 転がしチャレンジの判断・LINE・台帳、毎晩のデータ取り込み、紙上検証 edge-head-v1 | notes/bets/rolling-*.md、notes/bets/edge-head-v1.md |
| 穴狙い | 「展開が読めて、オッズに妙味があるレース」の厳選条件・パターン探し。1,000円で合成20〜50倍 | notes/chats/ana.md、notes/patterns.md |
| G1・SG（遊び） | G1・SG のモーター予想、節の資料、直前の展開予想とその検証 | notes/chats/g1sg.md、notes/reads/ |

共通の約束: 返事は日本語で短く。舟券は買わない（買うのは運用者）。トークンやIDをチャットに貼らない。rules.yaml は運用者の判断なしに変えない。LINE は「判断材料」。PR は頼まれたときだけ。気づき・決定は notes/ に書いてコミット（ファイルに残っているものだけが記憶）。
データ: data/boatrace.db は無ければ db.py が Parquet から組み立てる（時間がかかる）。締切時オッズ（odds）と直前情報（beforeinfo）は Parquet に入っている分まで。

## 司令塔の形（2026-11-14 追加、運用者の希望）
運用者はチャットを切り替えずに、**元のチャット（転がし）＝司令塔**とだけ話す。司令塔が話題に応じて「穴狙い」「G1・SG」のチャットへ依頼を送り（send_message）、各チャットは結果を notes/ に書いてプッシュし、要点を司令塔へ send_message で返す。司令塔がまとめて運用者に報告する。運用者が各チャットに直接書いた場合は、そのチャットが普通に答える。
