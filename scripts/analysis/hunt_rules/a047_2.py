import pandas as pd
def bets(df, odds=None):
    """1号艇がB1級・全国勝率4.8以上・展示タイムが他5艇の最速より0.02秒以上速い → 単勝1を100円"""
    oth = df[[f"et{k}" for k in range(2, 7)]].min(axis=1)
    m = (df.gr1 == 2) & (df.nw1 >= 4.8) & ((oth - df.et1) >= 0.02)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
