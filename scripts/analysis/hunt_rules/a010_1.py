import numpy as np, pandas as pd
def bets(df, odds=None):
    """展示タイム1位が1号艇で、2位との差が0.08秒以上、かつ1号艇の全国勝率が6艇中3位以下のとき、単勝1を100円。"""
    et = df[[f"et{k}" for k in range(1, 7)]].astype(float).values
    et = np.where(et > 0, et, np.nan)
    ok = ~np.isnan(et).any(axis=1)
    e2 = np.where(np.isnan(et), 99, et)
    top = np.argmin(e2, axis=1) + 1
    s = np.sort(e2, axis=1)
    gap = s[:, 1] - s[:, 0]
    nwr1 = df[[f"nw{k}" for k in range(1, 7)]].rank(axis=1, ascending=False, method="min")["nw1"].values
    m = ok & (top == 1) & (gap >= 0.08) & (nwr1 >= 3)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
