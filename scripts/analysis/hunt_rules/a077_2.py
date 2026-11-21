import pandas as pd
def bets(df, odds=None):
    """1号艇がB1で当地勝率6.0以上かつ当地勝率>=全国勝率（地元扱い）のレースで、単勝1を100円"""
    m = (df.gr1 == 2) & (df.lw1 >= 6.0) & (df.lw1 >= df.nw1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
