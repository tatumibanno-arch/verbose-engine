import pandas as pd
def bets(df, odds=None):
    """締切時オッズで1号艇の市場シェア ph1 が 0.20〜0.45（1が売れていない）、
    1号艇が展示タイム1位（er1==1）、1号艇がB級（gr1<=2）のレースで、単勝1を100円。"""
    m = (df.ph1 > 0.20) & (df.ph1 < 0.45) & (df.er1 == 1) & (df.gr1 <= 2)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
