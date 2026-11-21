import pandas as pd
def bets(df, odds=None):
    """1号艇がB1で、全国2連率35%以上なのに当地2連率が全国より15ポイント以上低い（当地出走あり）レース。1号艇の単勝100円。"""
    m = (df.gr1 == 2) & (df.n21 >= 35) & (df.l21 > 0) & ((df.l21 - df.n21) <= -15)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
