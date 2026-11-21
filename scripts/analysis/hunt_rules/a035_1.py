import pandas as pd
# 無風（風速1m以下・近似）で、1号艇がB1かつ展示タイムが単独最速 → 単勝1 を100円
def bets(df, odds=None):
    gap = df[[f"et{k}" for k in range(2, 7)]].min(axis=1) - df.et1
    m = (df.wind <= 1) & (df.gr1 == 2) & (gap > 0)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "1"; x["amount"] = 100
    return x
