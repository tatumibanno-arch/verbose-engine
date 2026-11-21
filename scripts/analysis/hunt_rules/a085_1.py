import pandas as pd, numpy as np
def bets(df, odds=None):
    """6号艇の展示タイムが単独最速で2番手より0.06秒以上速く、1号艇の全国勝率5.5以上 -> 2連単 1-6 を100円"""
    ET = df[[f"et{k}" for k in range(1, 7)]].to_numpy(dtype=float)
    ok = np.isfinite(ET).all(1) & (ET > 0).all(1)
    ET = np.where(ok[:, None], ET, np.inf)
    s = np.sort(ET, 1)
    best = ET.argmin(1) + 1
    gap = s[:, 1] - s[:, 0]
    m = ok & (best == 6) & (gap >= 0.06 - 1e-9) & (df.nw1.to_numpy(dtype=float) >= 5.5)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirensho"; r["combination"] = "1-6"; r["amount"] = 100
    return r
