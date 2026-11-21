import pandas as pd
def bets(df, odds=None):
    """2号艇がB1(gr=2)なのに展示1位、かつ全国勝率5.0以上 -> 単勝2を100円"""
    m = (df.gr2 == 2) & (df.er2 == 1) & (df.nw2 >= 5.0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "2"; r["amount"] = 100
    return r
