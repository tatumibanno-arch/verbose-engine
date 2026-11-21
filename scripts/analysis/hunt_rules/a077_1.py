import pandas as pd
def bets(df, odds=None):
    """1号艇がB級(B1/B2)で当地勝率5.5以上（当地で強い＝地元扱い）のレースで、単勝1を100円"""
    m = (df.gr1 <= 2) & (df.lw1 >= 5.5)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
