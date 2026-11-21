import numpy as np, pandas as pd
# 展示タイム上位2艇（同タイムは内枠優先）が1号艇と2号艇で、両者ともB1級のレースで、2連複 1-2 を100円
def bets(df, odds=None):
    ET = df[[f"et{k}" for k in range(1, 7)]].values.astype(float)
    o = np.argsort(ET, axis=1, kind="stable")
    a = o[:, 0] + 1; b = o[:, 1] + 1
    m = (np.minimum(a, b) == 1) & (np.maximum(a, b) == 2) & (df.gr1 == 2).values & (df.gr2 == 2).values & ~np.isnan(ET).any(axis=1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirenfuku"; r["combination"] = "1-2"; r["amount"] = 100
    return r
