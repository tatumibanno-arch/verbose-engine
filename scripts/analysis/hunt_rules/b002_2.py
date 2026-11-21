import pandas as pd
def bets(df, odds=None):
    """締切時オッズで ph1 が 0.20〜0.45、1号艇の展示タイムが他5艇の最速より厳密に速い（差>0）レースで、単勝1を100円。"""
    gap = df[[f"et{k}" for k in range(2, 7)]].min(axis=1) - df.et1
    m = (df.ph1 > 0.20) & (df.ph1 < 0.45) & (gap > 0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
