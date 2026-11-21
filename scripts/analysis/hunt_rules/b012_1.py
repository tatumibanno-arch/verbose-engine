import pandas as pd
# b012_1: 3連単の締切時オッズから、市場の2着・3着の付け方が Harville（単勝シェアから組み立てた確率）より
# 低く見ている目（= 2着・3着が過小評価）を拾う。1号艇頭、オッズ30倍以下、市場確率/Harville <= 0.8 の目を各100円。
ODDS_MAX = 30.0
HR_MAX = 0.8
HEAD = 1
def bets(df, odds=None):
    o = odds[["hd", "jcd", "rno", "cb", "odds"]].copy()
    o = o[o.odds > 0]
    ab = o.cb.str.split("-", expand=True).astype(int)
    o["a"], o["b"], o["c"] = ab[0], ab[1], ab[2]
    o["iv"] = 1 / o.odds
    k = ["hd", "jcd", "rno"]
    o["q"] = o.iv / o.groupby(k).iv.transform("sum")
    s1 = o.groupby(k + ["a"]).q.sum().rename("s1").reset_index()
    for col, nm in [("a", "sa"), ("b", "sb"), ("c", "sc")]:
        o = o.merge(s1.rename(columns={"a": col, "s1": nm}), on=k + [col], how="left")
    harv = o.sa * o.sb / (1 - o.sa) * o.sc / (1 - o.sa - o.sb)
    o["hr"] = o.q / harv
    o = o[(o.a == HEAD) & (o.odds <= ODDS_MAX) & (o.hr <= HR_MAX)]
    r = df[k].drop_duplicates()
    o = o.merge(r, on=k)
    out = o[k].copy()
    out["ticket_type"] = "sanrensho"; out["combination"] = o.cb.values; out["amount"] = 100
    return out.reset_index(drop=True)
