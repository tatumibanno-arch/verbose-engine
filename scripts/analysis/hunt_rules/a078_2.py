import pandas as pd
def bets(df, odds=None):
    """2号艇がB1なのに全国勝率5.25以上、かつ1号艇の全国勝率が5.0未満（級と勝率の食い違い：2号艇が実力上位）→ 単勝2を100円"""
    m = (df.gr2 == 2) & (df.nw2 >= 5.25) & (df.nw1 < 5.0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "2"; r["amount"] = 100
    return r
