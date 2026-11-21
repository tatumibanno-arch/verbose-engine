import numpy as np, pandas as pd
def bets(df, odds=None):
    """節の4日目以降。1号艇がB級(B1/B2)で、今節の平均着順(mf, 3走以上)が他5艇の最良より0.2以上良く、
    2〜6号艇にA級(A1/A2)が1人以上いるレース。単勝1を100円。"""
    d = df[df.day_no >= 4]
    others = [d[f"mf{k}"].where(d[f"mn{k}"] >= 3) for k in range(2, 7)]
    omin = pd.concat(others, axis=1).min(axis=1)
    m = (d.mn1 >= 3) & omin.notna() & (omin - d.mf1 >= 0.2) & (d.gr1 <= 2) \
        & (d[[f"gr{k}" for k in range(2, 7)]].max(axis=1) >= 3)
    r = d.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
