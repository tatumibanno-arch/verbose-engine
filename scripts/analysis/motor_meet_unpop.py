"""節間で好調なモーターに、売れていない選手が乗ったとき、頭（1着）で買うと得か。
    python scripts/analysis/motor_meet_unpop.py
売れ方は3連単の締切時オッズから出した頭シェア（実際に買える時点のオッズとは違う＝甘い）。"""
import sqlite3, pandas as pd, numpy as np
from pathlib import Path
c = sqlite3.connect(str(Path(__file__).resolve().parent.parent.parent / "data/boatrace.db"))
F = "20240101"
e = pd.read_sql(f"select e.hd,e.jcd,e.rno,e.lane,e.motor_no,e.grade,r.position from entries e join results r using(hd,jcd,rno,lane) where e.hd>='{F}'", c)
t = pd.read_sql(f"select hd,jcd,rno,combination,payout from payouts where hd>='{F}' and ticket_type='tansho'", c)
o = pd.read_sql("select hd,jcd,rno,substr(combination,1,1) h, sum(1.0/odds) q from odds where ticket_type='sanrensho' and odds>0 group by 1,2,3,4", c)
o["lane"] = o.h.astype(int); tot = o.groupby(["hd", "jcd", "rno"]).q.transform("sum"); o["share"] = o.q / tot; o["syn"] = 1 / o.q
e["pos"] = pd.to_numeric(e.position, errors="coerce"); e["d"] = pd.to_datetime(e.hd)
days = e[["jcd", "d"]].drop_duplicates().sort_values(["jcd", "d"])
days["meet"] = (days.groupby("jcd").d.diff().dt.days != 1).astype(int).groupby(days.jcd).cumsum()
e = e.merge(days, on=["jcd", "d"]); e["day_no"] = e.groupby(["jcd", "meet"]).d.rank(method="dense")
e["top2"] = (e.pos <= 2).astype(float); e["win"] = (e.pos == 1).astype(float)
dl = e.groupby(["jcd", "meet", "motor_no", "day_no"]).agg(n=("top2", "size"), t2=("top2", "sum")).reset_index().sort_values(["jcd", "meet", "motor_no", "day_no"])
g = dl.groupby(["jcd", "meet", "motor_no"]); dl["pn"] = g.n.cumsum() - dl.n; dl["pt2"] = g.t2.cumsum() - dl.t2
e = e.merge(dl[["jcd", "meet", "motor_no", "day_no", "pn", "pt2"]], on=["jcd", "meet", "motor_no", "day_no"])
e["r2"] = e.pt2 / e.pn.where(e.pn > 0)
t["lane"] = pd.to_numeric(t.combination, errors="coerce")
e = e.merge(t[["hd", "jcd", "rno", "lane", "payout"]], on=["hd", "jcd", "rno", "lane"], how="left")
e = e.merge(t[["hd", "jcd", "rno"]].drop_duplicates(), on=["hd", "jcd", "rno"])
e["ret"] = e.payout.fillna(0) / 100
e = e.merge(o[["hd", "jcd", "rno", "lane", "share", "syn"]], on=["hd", "jcd", "rno", "lane"], how="left")
e["ret3"] = e.win * e.syn  # 頭固定の3連単20点を合成買いしたときの戻り（締切時オッズ）
e["ym"] = e.hd.str[:6]
def show(name, m, col="ret"):
    x = e[m & e[col].notna()]
    if len(x) < 30: print(f"{name}: {len(x)}走（少なすぎ）"); return
    mo = x.groupby("ym")[col].mean()
    print(f"{name}: {len(x)}走 1着{x.win.mean():.1%} 回収{x[col].mean():.2f} 勝ち月{(mo>1).sum()}/{len(mo)} 最悪月{mo.min():.2f}")
hot = (e.pn >= 3) & (e.r2 >= .67); hot2 = (e.pn >= 4) & (e.r2 >= .8)
print("== 単勝。売れてない＝級別（2024-01〜、前夜に分かる） ==")
for nm, h in [("好調(3走67%)", hot), ("好調(4走80%)", hot2)]:
    show(f"{nm}・B1/B2", h & e.grade.isin(["B1", "B2"]))
    show(f"{nm}・B1/B2・2〜6号艇", h & e.grade.isin(["B1", "B2"]) & (e.lane >= 2))
    show(f"{nm}・B1/B2・4〜6号艇", h & e.grade.isin(["B1", "B2"]) & (e.lane >= 4))
print("== 売れてない＝3連単の頭シェア（締切時オッズ、2025-03〜） ==")
hs = e.share.notna()
show("基準: 頭シェア5%未満の全艇（単勝）", hs & (e.share < .05)); show("基準: 5〜10%（単勝）", hs & (e.share >= .05) & (e.share < .10)); show("基準: 10〜20%（単勝）", hs & (e.share >= .10) & (e.share < .20))
for nm, h in [("好調(3走67%)", hot), ("好調(4走80%)", hot2)]:
    for lo, hi in [(0, .05), (.05, .10), (.10, .20)]:
        m = h & hs & (e.share >= lo) & (e.share < hi)
        show(f"{nm}・頭シェア{lo:.0%}〜{hi:.0%}（単勝）", m); show(f"{nm}・頭シェア{lo:.0%}〜{hi:.0%}（3連単頭固定）", m, "ret3")
show("基準: 頭シェア5〜20%の全艇（3連単頭固定）", hs & (e.share >= .05) & (e.share < .20), "ret3")
