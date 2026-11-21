import numpy as np, pandas as pd
def bets(df, odds=None):
    """1号艇のモーター2連率が6艇中1位で2位との差10pt以上、かつ1号艇がA1以外 -> 単勝1を100円"""
    mo = df[[f"mo{k}" for k in range(1, 7)]].to_numpy(dtype=float)
    s = np.sort(np.nan_to_num(mo, nan=-1), 1)
    m = (np.nan_to_num(mo[:, 0], nan=-1) == s[:, -1]) & (s[:, -1] - s[:, -2] >= 10) & (df.gr1.to_numpy() <= 3)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
