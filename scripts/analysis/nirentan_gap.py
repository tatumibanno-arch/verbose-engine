"""notes/analysis/nirentan_gap_plan.md で事前に固定した6通りを、確定オッズで検証する。"""
import sqlite3, numpy as np, pandas as pd
con = sqlite3.connect("data/boatrace.db", timeout=600)
LO, HI = "20251101", "20261031"
o = pd.read_sql("select hd,jcd,rno,ticket_type t,combination c,odds from odds where hd between ? and ? and ticket_type in ('nirensho','sanrensho') and odds>0", con, params=(LO, HI))
pay = pd.read_sql("select hd,jcd,rno,combination c,payout from payouts where ticket_type='nirensho' and hd between ? and ?", con, params=(LO, HI)).drop_duplicates(["hd","jcd","rno"])
o["inv"] = 1 / o.odds
k = ["hd","jcd","rno"]
both = o.groupby(k).t.nunique(); both = both[both == 2].index
o = o.set_index(k).loc[both].reset_index()
o["q"] = o.inv / o.groupby(k + ["t"]).inv.transform("sum")
t2 = o[o.t == "nirensho"][k + ["c","q","odds"]].rename(columns={"q":"q2"})
t3 = o[o.t == "sanrensho"].copy(); t3["c"] = t3.c.str[:3]
p3 = t3.groupby(k + ["c"]).q.sum().rename("p3").reset_index()
d = t2.merge(p3, on=k + ["c"]); d["ratio"] = d.p3 / d.q2
d = d.merge(pay.rename(columns={"c":"win"}), on=k, how="left")
d["ret"] = np.where(d.c == d.win, d.payout, 0)
d["m"] = d.hd.str[:6]
print("races", d.groupby(k).ngroups, "months", d.m.nunique())
C = [(1,1.3,0.08,1),(2,1.4,0.08,1),(3,1.6,0.08,1),(4,2.0,0.08,1),(5,1.6,0.15,1),(6,1.6,0.08,2)]
DISC = [m for m in sorted(d.m.unique()) if m <= "202604"]; VAL = [m for m in sorted(d.m.unique()) if m >= "202605"]
rows = []
for no, r, pmin, kmax in C:
    s = d[(d.ratio >= r) & (d.p3 >= pmin)].sort_values("ratio", ascending=False).groupby(k).head(kmax)
    for lab, ms in [("発見", DISC), ("検証", VAL), ("全期間", DISC + VAL)]:
        x = s[s.m.isin(ms)]
        mm = x.groupby("m").agg(b=("ret","size"), r=("ret","sum")); mr = mm.r / (mm.b * 100)
        days = d[d.m.isin(ms)].hd.nunique()
        rows.append(dict(候補=no, ずれ=r, 確率下限=pmin, 点=kmax, 期間=lab, 一日点=round(len(x)/days,1), 的中=int((x.ret>0).sum()),
                         回収率=round(x.ret.sum()/(len(x)*100),3), 勝ち月=f"{int((mr>1).sum())}/{len(mr)}", 最悪月=round(mr.min(),2)))
print(pd.DataFrame(rows).to_string(index=False))
