import pandas as pd, numpy as np
def bets(df, odds=None):
    """5号艇か6号艇の展示タイムが単独最速で2番手より0.07秒以上速い -> 2連単 1-(その艇) を100円"""
    ET = df[[f"et{k}" for k in range(1, 7)]].to_numpy(dtype=float)
    ok = np.isfinite(ET).all(1) & (ET > 0).all(1)
    ET = np.where(ok[:, None], ET, np.inf)
    s = np.sort(ET, 1)
    best = ET.argmin(1) + 1
    gap = s[:, 1] - s[:, 0]
    m = ok & np.isin(best, [5, 6]) & (gap >= 0.07 - 1e-9)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirensho"; r["combination"] = "1-" + pd.Series(best[m], index=r.index).astype(str); r["amount"] = 100
    return r
