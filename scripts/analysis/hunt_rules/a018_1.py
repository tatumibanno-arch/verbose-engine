import numpy as np, pandas as pd
def bets(df, odds=None):
    """節の4日目以降。1号艇がB級(B1/B2)で、今節の平均着順(mf, 3走以上)が6艇で最良(同値含む)、
    かつ2〜6号艇にA1が1人以上いるレース。単勝1を100円。"""
    d = df[df.day_no >= 4]
    others = []
    for k in range(2, 7):
        others.append(d[f"mf{k}"].where(d[f"mn{k}"] >= 3))
    omin = pd.concat(others, axis=1).min(axis=1)
    m = (d.mn1 >= 3) & (d.mf1 <= omin) & omin.notna() & (d.gr1 <= 2) \
        & (d[[f"gr{k}" for k in range(2, 7)]].max(axis=1) == 4)
    r = d.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
