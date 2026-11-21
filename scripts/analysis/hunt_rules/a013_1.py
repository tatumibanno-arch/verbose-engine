import numpy as np, pandas as pd
# 展示タイム上位2艇（同タイムは内枠優先）がともに1〜3号艇、かつ若松(20)・福岡(22)のレースで、その2艇の2連複を100円
def bets(df, odds=None):
    ET = df[[f"et{k}" for k in range(1, 7)]].values.astype(float)
    o = np.argsort(ET, axis=1, kind="stable")
    a = o[:, 0] + 1; b = o[:, 1] + 1
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    m = df.jcd.astype(int).isin([20, 22]).values & (hi <= 3) & ~np.isnan(ET).any(axis=1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirenfuku"
    r["combination"] = [f"{x}-{y}" for x, y in zip(lo[m], hi[m])]
    r["amount"] = 100
    return r
