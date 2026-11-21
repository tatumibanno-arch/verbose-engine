import pandas as pd
def bets(df, odds=None):
    """展示タイム: 1号艇が他艇の最速より0.20秒以上遅い（1位との差が大きい）レースで、単勝2を100円。展示（締切前情報）を使う。"""
    fastest_other = df[[f"et{k}" for k in range(2, 7)]].min(axis=1)
    m = (df.et1 - fastest_other) >= 0.20
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "2"; r["amount"] = 100
    return r
