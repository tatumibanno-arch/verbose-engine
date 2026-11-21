import numpy as np
import pandas as pd
def bets(df, odds=None):
    """波5cm以上（近似：レース時記録）で、展示タイム最速艇（同タイムは内側）が3号艇か4号艇、
    かつその艇がA1でない(級<=A2)レースで、その艇の単勝を100円。"""
    E = df[[f"et{k}" for k in range(1, 7)]].astype(float).values
    ok = ~np.isnan(E).all(axis=1)
    E2 = np.where(np.isnan(E), 99.0, E)
    b = E2.argmin(axis=1)
    G = df[[f"gr{k}" for k in range(1, 7)]].astype(float).values
    bg = G[np.arange(len(df)), b]
    m = ok & (df.wave.astype(float).values >= 5) & np.isin(b + 1, [3, 4]) & (bg <= 3)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"
    r["combination"] = (b[m] + 1).astype(str)
    r["amount"] = 100
    return r
