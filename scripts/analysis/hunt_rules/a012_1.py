import pandas as pd
def bets(df, odds=None):
    """1号艇がB1(gr=2)なのに展示1位、かつ全国勝率4.8以上 -> 単勝1を100円"""
    m = (df.gr1 == 2) & (df.er1 == 1) & (df.nw1 >= 4.8)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
