"""par_aff: 当地相性・勢い・疲労・地元の検証用に、全期間の出走+着順+3連単結果を1回だけ読んでParquetに保存。"""
import sqlite3, pandas as pd, sys
out = sys.argv[1]
c = sqlite3.connect('file:data/boatrace.db?mode=ro', uri=True)
J = lambda k: f"json_extract(data,'$.{k}')"
e = pd.read_sql(f"select hd,jcd,rno,lane,cast(racer_no as int) racer,grade,{J('branch')} branch,{J('national_win_rate')} nw,{J('national_2ren_rate')} n2 from entries", c)
r = pd.read_sql("select hd,jcd,rno,lane,position pos from results", c)
p = pd.read_sql("select hd,jcd,rno,combination comb,payout from payouts where ticket_type='sanrensho'", c)
e.to_parquet(out+'/e.parquet'); r.to_parquet(out+'/r.parquet'); p.to_parquet(out+'/p.parquet')
print(len(e), len(r), len(p))
