"""締切6分前のオッズ（data/live/log の *_odds.csv）と確定オッズ（odds テーブル）を比べる。
使い方: python scripts/analysis/live_vs_close.py 20261103"""
import sqlite3, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from common import VENUES
hd = sys.argv[1]
con = sqlite3.connect("data/boatrace.db", timeout=600)
snap = pd.concat([pd.read_csv(p, dtype={"jcd": str, "combination": str}) for p in glob.glob(f"data/live/log/{hd}_s*_odds.csv")])
snap["jcd"] = snap.jcd.str.zfill(2)
fin = pd.read_sql("select jcd, rno, ticket_type, combination, odds from odds where hd=?", con, params=(hd,))
pay = pd.read_sql("select jcd, rno, ticket_type, combination, payout from payouts where hd=?", con, params=(hd,))
m = snap.merge(fin, on=["jcd", "rno", "ticket_type", "combination"], suffixes=("_6", "_f"))
m = m[(m.odds_6 > 0) & (m.odds_f > 0)]
def probs(d):
    d = d.copy(); d["inv"] = 1 / d.odds
    d["q"] = d.inv / d.groupby(["jcd", "rno", "ticket_type"]).inv.transform("sum")
    return d
def pair_p3(d):
    t3 = d[d.ticket_type == "sanrensho"].copy(); t3["pair"] = t3.combination.str[:3]
    return t3.groupby(["jcd", "rno", "pair"]).q.sum().rename("p3").reset_index()
out = []
print(f"## {hd} 締切6分前 → 確定 のオッズの動き（{m.groupby(['jcd','rno']).ngroups}レース）")
for tt, name in [("tansho", "単勝"), ("nirensho", "2連単"), ("sanrensho", "3連単")]:
    x = m[m.ticket_type == tt]; r = x.odds_f / x.odds_6
    print(f"- {name}: 確定/6分前 中央値 {r.median():.2f}、下がった割合 {(r < 1).mean():.0%}、半分以下になった割合 {(r <= 0.5).mean():.0%}")
# 6分前・確定それぞれで2連単のずれ（3連単由来/2連単由来）
res = {}
for lab, col in [("6", "odds_6"), ("f", "odds_f")]:
    d = probs(m.rename(columns={col: "odds"})[["jcd", "rno", "ticket_type", "combination", "odds"]])
    t2 = d[d.ticket_type == "nirensho"][["jcd", "rno", "combination", "q", "odds"]].rename(columns={"combination": "pair"})
    t2 = t2.merge(pair_p3(d), on=["jcd", "rno", "pair"]); t2["ratio"] = t2.p3 / t2.q
    res[lab] = t2.set_index(["jcd", "rno", "pair"])
j = res["6"].join(res["f"], lsuffix="_6", rsuffix="_f")
p2 = pay[pay.ticket_type == "nirensho"].rename(columns={"combination": "pair"})[["jcd", "rno", "pair", "payout"]]
j = j.reset_index().merge(p2, on=["jcd", "rno", "pair"], how="left"); j["ret"] = j.payout.fillna(0)
sel = j[(j.ratio_6 >= 1.4) & (j.p3_6 >= 0.08)]
print(f"\n## 6分前にずれ≥1.4 だった2連単（{len(sel)}点、通知の条件と同じ・1レース2点の上限なし）")
print(f"- オッズ 6分前 中央値 {sel.odds_6.median():.1f}倍 → 確定 {sel.odds_f.median():.1f}倍（確定/6分前 中央値 {(sel.odds_f/sel.odds_6).median():.2f}）")
print(f"- ずれ 6分前 中央値 {sel.ratio_6.median():.2f} → 確定 {sel.ratio_f.median():.2f}（確定でも≥1.4 のまま {(sel.ratio_f>=1.4).mean():.0%}）")
print(f"- 的中 {int((sel.ret>0).sum())}/{len(sel)}、確定の払戻での回収率 {sel.ret.sum()/(len(sel)*100):.2f}")
print(f"- 6分前のオッズで計算した期待値の平均 {(sel.p3_6*sel.odds_6).mean():.2f} → 確定オッズで計算 {(sel.p3_f*sel.odds_f).mean():.2f}")
al = pd.read_csv(glob.glob(f"data/live/log/{hd}_s*_2t.csv")[0], dtype={"jcd": str}); al["jcd"] = al.jcd.str.zfill(2)
al = al.rename(columns={"combination": "pair"}).merge(j[["jcd", "rno", "pair", "odds_f", "ratio_f", "ret"]], on=["jcd", "rno", "pair"], how="left")
print("\n## 実際に LINE に流した2連単")
for r in al.itertuples():
    print(f"- {r.close} {VENUES.get(r.jcd)}{r.rno}R {r.pair}: {r.odds}倍 → 確定 {r.odds_f}倍、ずれ {r.ratio} → {r.ratio_f:.2f}、{'◎' + str(int(r.ret)) + '円' if r.ret > 0 else '×'}{'' if r.sent else '（時間外で未送信）'}")
s = al[al.sent == 1]
print(f"- 送信 {len(s)}点: 的中 {int((s.ret>0).sum())}、購入 {len(s)*100}円、払戻 {int(s.ret.sum())}円、回収率 {s.ret.sum()/(len(s)*100):.2f}")
