import pandas as pd
def bets(df, odds=None):
    """1号艇が当地巧者（当地勝率-全国勝率>=2.0、かつ当地勝率がレース内1位）のとき単勝1を100円"""
    lw = df[[f"lw{k}" for k in range(1, 7)]]
    m = (df.lw1 > 0) & ((df.lw1 - df.nw1) >= 2.0) & (df.lw1 >= lw.max(axis=1))
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
