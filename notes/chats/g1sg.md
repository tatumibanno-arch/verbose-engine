# G1・SG（遊び）チャットの引き継ぎ（2026-11-14）
- 運用者が「遊び」で楽しむ用。G1・SG のモーター予想、節の資料、直前の展開予想、その検証。
- 過去にやったこと: 鳴門G1（2026-11）・住之江の節の資料（scripts/analysis/meet_dossier.py、notes/reads/）、展示を見てからの「万舟」「中穴」の型（notes/learnings.md 2026-11-06〜07。12レース65点で回収2.10・的中2、1日分なので評価しない）、前検の印の当たり方。
- 型: 締切25分前に展示タイム・スタ展・チルト・風を取り、前夜に整理した展開パターンのどれが濃いかを決め、本線2〜3点＋ヒネリ2〜3点に今のオッズを付けて出す。住之江は展示が締切4〜8分前にしか出ない。
- 直前の情報は scripts/scrape_web.py（beforeinfo、odds3t）で取る。LINE に送るときは notes/line/ にファイルを書いてプッシュし、GitHub Actions の live_alert.yml（ref main、inputs: mode=send, file=...）で送って成功を確認する。文面は「判断材料」。
- 記録は残す（当たり外れ、見送りの判断も）。少ない本数で良し悪しを断定しない。

## 2026-11-14 びわこ大賞・丸亀京極賞の資料（司令塔からの依頼）
- 事実確認（公式 boatrace.jp のレース一覧）: びわこ G1「開設74周年記念 びわこ大賞」11/14〜11/19（初日は 11/14。運用者の記憶「明日から」より1日早い）。丸亀 G1「京極賞 開設74周年記念競走」11/15〜11/20、ナイター。ほかに 11/14〜15 の G1・SG は無し。
- 資料: `notes/reads/biwako-taisho-2026-11.md`、`notes/reads/marugame-kyogokusho-2026-11.md`。作るのは `scripts/analysis/g1_prep.py`（前検・モーター・選手・場の傾向・節間の気配・翌日の番組を1本で出す。`meet_dossier.py` の後継）。
- 前検は公式 `race/rankingmotor?jcd=..&hd=初日` にある（モーター抽選・前検タイム・記者コメント）。取れたら `notes/reads/sources/zenken_{jcd}_{初日}.json` に保存。出る前は `race/assen`（出場予定レーサー一覧）で選手だけ分かる。丸亀は 11/14 朝の時点で前検まだ（確認できず）。丸亀の場の公式サイトはこの環境から届かない。
- 丸亀の現モーターは 2026/09/17 から（1機あたり30走前後）。数字が少ないのでモーターの上下は目安どまり。びわこは 2026/04/08 から（1機140走前後）。
- 期別成績は 2026/04/30 までの期が最新（fan2610 はまだ公式に無い）。決まり手・F持ちは手元のデータに無い（決まり手は公式の場データを `notes/reads/sources/stadium_{jcd}.md` に写した）。
- 節間の更新（毎晩、その日の全レース後）:
  1. `git pull --rebase --autostash origin claude/tender-newton-ymc8j0`
  2. `python scripts/ingest.py performance --date 当日`（成績）→ `python scripts/ingest.py schedule --date 翌日`（番組表）→ `python scripts/scrape_web.py --date 当日 --pages beforeinfo`（チルト・部品交換）
  3. `python scripts/analysis/g1_prep.py --jcd 11 --from 20261114 --to 20261119 --title "…" --out notes/reads/biwako-taisho-2026-11.md`、丸亀は `--jcd 15 --from 20261115 --to 20261120 --out notes/reads/marugame-kyogokusho-2026-11.md`
  4. notes/ だけコミットしてプッシュ。Parquet の書き出しは転がしチャットの夜の処理に任せる（同じファイルを2つのチャットが書くとぶつかるため）。
  - 丸亀の前検は 11/14 の夕方以降に 3 を回せば入る（選手ごとのモーターが決まる）。
- 環境メモ: 新しいコンテナでは `pip install pyarrow boatrace-lzh` が要る。`data/boatrace.db` の組み立ては約2分（9GB）。Parquet は成績 11/08 までだったので、11/09〜13 の成績と 11/10〜14 の番組表はこのチャットの作業用DBにだけ取り込んだ（Parquet には書いていない）。
- **まっさらなコンテナで節間の更新を回すときの前準備**（上の手順 2 の前に必ず）:
  - `pip install pyarrow boatrace-lzh pyyaml`、続けて `python scripts/db.py stats`（DB を Parquet から組み立てる。約2分。出た `results` と `entries` の `to` の日付を見る）。
  - Parquet に無い日を埋める: `python scripts/ingest.py performance --from <results の to の翌日> --to 当日`、`python scripts/ingest.py schedule --from <entries の to の翌日> --to 翌日`（11/14 朝の時点では成績 11/08・番組表 11/09 までだったので、`--from 20261109` と `--from 20261110`）。これを抜くと、モーターの直近成績と節間の章（番組表と成績の突き合わせ）が欠ける。取り直しは上書きなので重なっても害はない。翌日の番組表は前夜に出る（まだなら 404 で 0 件、それでよい）。
  - `scrape_web.py` は成績を取り込んだ日のレースしか見に行かないので、成績の取り込みの後に回す。
