import pandas as pd
# 1号艇が売れなさすぎる場（探す期間で1号艇の勝率/市場シェア比が高かった8場）で、
# 1号艇の単勝シェアが0.25〜0.65、かつ展示タイム1位なら 単勝1 を100円。
TOP = [18, 13, 5, 8, 19, 16, 7, 22]
def bets(df, odds=None):
    m = df.jcd.isin(TOP) & df.ph1.between(0.25, 0.65) & (df.er1 == 1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
