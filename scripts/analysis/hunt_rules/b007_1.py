import pandas as pd
def bets(df, odds=None):
    """1号艇が展示タイム1位（er1==1）、B級（gr1<=2）、締切時の1頭シェア ph1 が 0.2〜0.6（実力の割に売れていない）→ 単勝1を100円"""
    m = (df.er1 == 1) & (df.gr1 <= 2) & (df.ph1 > 0.2) & (df.ph1 < 0.6)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
