import pandas as pd
def bets(df, odds=None):
    """1号艇がB1なのに全国勝率5.25以上・モーター2連率33%以上（級に比べ実力が上の1号艇）→ 単勝1を100円"""
    m = (df.gr1 == 2) & (df.nw1 >= 5.25) & (df.mo1 >= 33)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
