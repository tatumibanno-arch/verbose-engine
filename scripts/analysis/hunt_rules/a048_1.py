import numpy as np, pandas as pd
def bets(df, odds=None):
    """1号艇の全国勝率が他5艇の最高より1.0超高く、展示タイム順位が2位以内なら 1号艇の複勝を100円"""
    nw = df[[f"nw{k}" for k in range(2, 7)]].max(axis=1)
    m = (df.nw1 - nw > 1.0) & (df.er1 <= 2)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "fukusho"; r["combination"] = "1"; r["amount"] = 100
    return r
