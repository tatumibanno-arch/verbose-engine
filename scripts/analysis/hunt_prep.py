"""総当たり検証（2026-11-21）の共通データを作る。1レース1行。SQLite から読む。
    python scripts/analysis/hunt_prep.py OUTDIR HOLDDIR
期間A（オッズを使わない条件）: 探す 2016-01〜2022-12 / 確かめる 2023-01〜2025-12 / 最後 2026-01〜2026-11
期間B（締切時オッズを使う条件）: 探す 2025-03〜2025-12 / 確かめる 2026-01〜2026-06 / 最後 2026-07〜2026-11
「最後」は HOLDDIR に分けて置き、探す側には見せない。
"""
import sqlite3, sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent
OUT, HOLD = Path(sys.argv[1]), Path(sys.argv[2])
c = sqlite3.connect(str(ROOT / "data/boatrace.db"))
F, T = "20151201", "20261130"
K = ["hd", "jcd", "rno"]
J = lambda k: f"json_extract(data,'$.{k}')"
en = pd.read_sql(f"""select hd,jcd,rno,lane,cast(racer_no as int) racer,grade,close_time,race_name,motor_no,
  {J('national_win_rate')} nw,{J('national_2ren_rate')} n2,{J('local_win_rate')} lw,{J('local_2ren_rate')} l2,
  {J('motor_2ren_rate')} mo,{J('boat_2ren_rate')} bo,{J('age')} age,{J('weight')} wt
  from entries where hd between '{F}' and '{T}'""", c)
rs = pd.read_sql(f"""select hd,jcd,rno,lane,position pos,{J('exhibition_time')} et,{J('entrance_position')} ent,{J('st_timing')} st
  from results where hd between '{F}' and '{T}'""", c)
ra = pd.read_sql(f"select hd,jcd,rno,distance,weather,wind_direction wdir,wind_speed wind,wave_height wave from races where hd between '{F}' and '{T}'", c)
print("読み込み", len(en), len(rs), len(ra), flush=True)
en["gr"] = en.grade.map({"A1": 4, "A2": 3, "B1": 2, "B2": 1})
d = en.merge(rs, on=K + ["lane"], how="inner")
d["dt"] = pd.to_datetime(d.hd)
# 節 = 同じ場で日付が連続する塊
days = d[["jcd", "dt"]].drop_duplicates().sort_values(["jcd", "dt"])
days["meet"] = (days.groupby("jcd").dt.diff().dt.days != 1).astype(int).groupby(days.jcd).cumsum()
days["day_no"] = days.groupby(["jcd", "meet"]).cumcount() + 1
d = d.merge(days, on=["jcd", "dt"])
# 節間（前日まで）: 選手の平均ST・平均着・走数、モーターの2連対率・走数
d["t2"] = (d.pos <= 2).astype(float); d["posf"] = d.pos.astype(float)
def prior(keys, cols):
    g = d.groupby(keys + ["day_no"]).agg(n=("lane", "size"), **{k: (v, "sum") for k, v in cols.items()}).reset_index().sort_values(keys + ["day_no"])
    gg = g.groupby(keys)
    for k in ["n"] + list(cols): g["p_" + k] = gg[k].cumsum() - g[k]
    return g[keys + ["day_no"] + ["p_n"] + ["p_" + k for k in cols]]
d["stv"] = d.st.where(d.st.notna(), 0.0); d["stn"] = d.st.notna().astype(float)
d["pfv"] = d.posf.fillna(6.0)
pr = prior(["jcd", "meet", "racer"], {"stv": "stv", "stn": "stn", "pfv": "pfv"})
pr["ms"] = pr.p_stv / pr.p_stn.where(pr.p_stn > 0); pr["mf"] = pr.p_pfv / pr.p_n.where(pr.p_n > 0); pr["mn"] = pr.p_n
d = d.merge(pr[["jcd", "meet", "racer", "day_no", "ms", "mf", "mn"]], on=["jcd", "meet", "racer", "day_no"], how="left")
pm = prior(["jcd", "meet", "motor_no"], {"t2": "t2"})
pm["mm"] = pm.p_t2 / pm.p_n.where(pm.p_n > 0); pm["mmn"] = pm.p_n
d = d.merge(pm[["jcd", "meet", "motor_no", "day_no", "mm", "mmn"]], on=["jcd", "meet", "motor_no", "day_no"], how="left")
# 期別成績（算出期間の最終日から61日以上たった最新の期）
st = pd.read_sql("select hd shd, cast(racer_no as int) racer, data from racer_stats where hd>='20150430'", c)
rows = []
for shd, racer, js in st.itertuples(index=False):
    j = json.loads(js)
    for k in range(1, 7):
        rows.append((shd, racer, k, j.get(f"c{k}_win_rate"), j.get(f"c{k}_top2_rate"), j.get(f"c{k}_avg_st"), j.get(f"c{k}_starts"), j.get("avg_st"), j.get("ability"), j.get("sex")))
cs = pd.DataFrame(rows, columns=["shd", "racer", "lane", "cw", "ct", "cst", "cn", "ast", "abil", "sex"])
shds = sorted(cs.shd.unique()); lim = {s: pd.to_datetime(s) + pd.Timedelta(days=61) for s in shds}
hds = d[["hd", "dt"]].drop_duplicates()
hds["shd"] = [max([s for s in shds if lim[s] <= t], default=None) for t in hds.dt]
d = d.merge(hds[["hd", "shd"]], on="hd").merge(cs, on=["shd", "racer", "lane"], how="left")
d["er"] = d.groupby(K).et.rank(method="min")
d = d[d.hd >= "20160101"]
vals = ["racer", "gr", "nw", "n2", "lw", "l2", "mo", "bo", "age", "wt", "sex", "cw", "ct", "cst", "cn", "ast", "abil", "ms", "mf", "mn", "mm", "mmn", "et", "er", "LEAK_ent", "LEAK_st", "LEAK_pos"]
d = d.rename(columns={"ent": "LEAK_ent", "st": "LEAK_st", "pos": "LEAK_pos"})
for v in vals: d[v] = pd.to_numeric(d[v], errors="coerce").astype("float32")
w = d.pivot_table(index=K, columns="lane", values=vals, dropna=False, aggfunc="first")
w.columns = [f"{a}{int(l)}" for a, l in w.columns]
race = d.groupby(K).agg(n=("lane", "size"), close_time=("close_time", "first"), race_name=("race_name", "first"), day_no=("day_no", "first")).reset_index()
race = race[race.n == 6].drop(columns="n").merge(ra, on=K, how="left").merge(w.reset_index(), on=K)
# 着順（1〜3着の艇番）
top = d[d.LEAK_pos.between(1, 3)].pivot_table(index=K, columns="LEAK_pos", values="lane", aggfunc="first")
top.columns = [f"w{int(x)}" for x in top.columns]
race = race.merge(top.reset_index(), on=K, how="inner")
# 市場: 締切時の3連単オッズから頭ごとのシェア
o = pd.read_sql("select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and odds>0", c)
o["q"] = 1 / o.odds; o["h"] = o.cb.str[0].astype(int)
H = o.pivot_table(index=K, columns="h", values="q", aggfunc="sum"); S = H.sum(axis=1)
H = H.div(S, axis=0); H.columns = [f"ph{i}" for i in H.columns]; H["nodds"] = o.groupby(K).size()
race = race.merge(H.reset_index(), on=K, how="left")
race["ym"] = race.hd.str[:6]; race["jcd"] = race.jcd.astype(int); o["jcd"] = o.jcd.astype(int)
py = pd.read_sql("select hd,cast(jcd as int) jcd,rno,ticket_type,combination,payout from payouts where hd between '20160101' and '20261130'", c)
OUT.mkdir(parents=True, exist_ok=True); HOLD.mkdir(parents=True, exist_ok=True)
def cut(df, a, b): return df[(df.hd >= a) & (df.hd <= b)]
P = {"A_disc": ("20160101", "20221231"), "A_valid": ("20230101", "20251231"), "A_hold": ("20260101", "20261130"),
     "B_disc": ("20250301", "20251231"), "B_valid": ("20260101", "20260630"), "B_hold": ("20260701", "20261130")}
for nm, (a, b) in P.items():
    dst = HOLD if nm.endswith("hold") else OUT
    r = cut(race, a, b)
    if nm[0] == "B": r = r[r.nodds >= 100]
    r.to_parquet(dst / f"{nm}_races.parquet"); cut(py, a, b).to_parquet(dst / f"{nm}_payouts.parquet")
    if nm[0] == "B": cut(o, a, b)[K + ["cb", "odds"]].to_parquet(dst / f"{nm}_odds3t.parquet")
    print(nm, len(r), r.hd.min(), r.hd.max(), flush=True)
print("列", list(race.columns)[:40], race.shape)
