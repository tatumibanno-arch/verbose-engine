import numpy as np, pandas as pd
def bets(df, odds=None):
    """2号艇のモーター2連率が6艇中1位で2位との差5pt以上、かつ2号艇の全国勝率が1号艇より1.0超高い -> 単勝2を100円"""
    mo = df[[f"mo{k}" for k in range(1, 7)]].to_numpy(dtype=float)
    s = np.sort(np.nan_to_num(mo, nan=-1), 1)
    m = (np.nan_to_num(mo[:, 1], nan=-1) == s[:, -1]) & (s[:, -1] - s[:, -2] >= 5) & ((df.nw2 - df.nw1).to_numpy() > 1.0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "2"; r["amount"] = 100
    return r
