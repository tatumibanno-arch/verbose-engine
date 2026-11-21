import pandas as pd
# 1号艇が売れなさすぎる場（探す期間で1号艇の勝率/市場シェア比が高かった8場）で、
# 1号艇がB級（B1/B2）、単勝シェア0.25〜0.65なら 単勝1 を100円（市場がB級のイン逃げを低く見る）。
TOP = [18, 13, 5, 8, 19, 16, 7, 22]
def bets(df, odds=None):
    m = df.jcd.isin(TOP) & df.ph1.between(0.25, 0.65) & (df.gr1 <= 2)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
