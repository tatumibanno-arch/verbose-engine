# 総当たり検証（2026-11-21）共通の決まり

目的: 競艇で「この条件のレースを、この買い方で買えば回収率が1.0を超える」ものを探す。偶然の当たりを拾わないための決まりを守ること。

## データ（/home/user/hunt/、1レース1行、6艇そろったレースだけ）
- 期間A（オッズを使わない条件）: `A_disc_races.parquet` 探す期間 2016-01〜2022-12（約37万レース）／ `A_valid_races.parquet` 確かめる期間 2023-01〜2025-12
- 期間B（締切時オッズを使う条件）: `B_disc_races.parquet` 2025-03〜2025-12 ／ `B_valid_races.parquet` 2026-01〜2026-06。`B_*_odds3t.parquet` は3連単の締切時オッズ（hd,jcd,rno,cb='1-2-3',odds）
- 払戻: `{期間}_payouts.parquet`（hd,jcd,rno,ticket_type,combination,payout=100円あたり）。券種: tansho fukusho nirensho nirenfuku wide sanrensho sanrenfuku。順不同の券種は小さい順（'1-3-4'）
- その後の期間（最後の確認用）は別の場所にあり、見てはいけない。data/boatrace.db や data/parquet も読まない（最後の期間が混ざるため）

### 列（k=1..6 は艇番＝枠）
- レース: hd(YYYYMMDD) jcd(場コード 1〜24) rno close_time('HH:MM') race_name day_no(節の何日目) distance ym
- 前夜に分かる: racer{k} gr{k}(級 4=A1,3=A2,2=B1,1=B2) nw{k}全国勝率 n2{k}全国2連率 lw{k}当地勝率 l2{k}当地2連率 mo{k}モーター2連率 bo{k}ボート2連率 age{k} wt{k}体重 sex{k}(1男2女)
  期別成績（その枠のコース別）: cw{k}1着率 ct{k}2連率 cst{k}平均ST cn{k}出走数、ast{k}全体平均ST abil{k}能力指数
  今節の前日まで: ms{k}平均ST mf{k}平均着順 mn{k}走数、mm{k}モーターの今節2連対率 mmn{k}モーター走数（初日は欠損）
- 締切前に分かる: et{k}展示タイム er{k}展示順位（1=最速）。weather wind wdir wave はレース時の記録（締切前の表示とほぼ同じとみなす。使ったら「近似」と書く）
- 締切時オッズ（期間Bだけ。実際に買える時点より後の情報で、当たり目のオッズは締切前より下がる＝この検証は甘い）: ph{k} 3連単オッズから出した k頭の市場シェア、nodds
- **使ってはいけない（結果そのもの）**: LEAK_ent{k}本番の進入 LEAK_st{k}本番ST LEAK_pos{k}着順、w1 w2 w3（1〜3着の艇番）。探す期間の分析で「答え」として見るのはよいが、条件には使えない（採点器は渡さない）

## 採点（必ずこれを使う）
`python /home/user/verbose-engine/scripts/analysis/hunt_eval.py ルール.py --period A_disc --agent 自分のID`
- ルールは `/home/user/hunt/rules/{自分のID}_名前.py` に置く。`def bets(df, odds=None)` が DataFrame[hd,jcd,rno,ticket_type,combination,amount] を返す（amount は100円単位）。例: `/home/user/hunt/rules/_example.py`
- 採点器は回収率・的中レース率・月ごとの勝ち月数・最悪月・「最大の1本を除いた回収率」・前半/後半の回収率を出す
- 何も工夫しないと回収率は 0.75 前後（単勝・複勝は0.80前後）。穴ほど低い（3連単300倍以上は約0.5）

## 手順（偶然を拾わないため厳守）
1. 探す期間（disc）だけで条件と買い方を探す。自分で pandas で集計してよい。採点器を disc で回すのは12回まで。
2. 最終ルールを最大2本に決める。条件は少なく単純に（しきい値は3つまで）。対象レースが disc で月30レース未満になるほど絞らない。
3. 決めたルールを確かめる期間（valid）で1回ずつ採点する（valid の採点は1体3回まで。採点器が止める）。valid の結果を見てルールを直さない。直したくなっても直さず、そのまま報告する。
4. disc で回収率1.0に届く条件が見つからなくても、いちばんましな1本を valid にかけて正直に報告する。無理に良く見せない。
5. リポジトリのファイルは変えない。git の操作もしない。作るのは /home/user/hunt/rules/ の自分のファイルだけ。メモリを食う処理（全列を何度も複製、12GB超）は避け、必要な列だけ読む。作業は15分以内を目安に切り上げる。
