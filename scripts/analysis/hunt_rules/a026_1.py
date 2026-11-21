import numpy as np, pandas as pd
def bets(df, odds=None):
    """1号艇の能力指数が60以上で、2〜6号艇の最大より10以上高い（実力差が大きい）レースで単勝1を100円"""
    A = df[[f"abil{k}" for k in range(1, 7)]].replace(0, np.nan)
    d1 = A["abil1"] - A[[f"abil{k}" for k in range(2, 7)]].max(axis=1)
    m = (A["abil1"] >= 60) & (d1 >= 10)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "1"; x["amount"] = 100
    return x
