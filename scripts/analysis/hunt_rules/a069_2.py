import pandas as pd
def bets(df, odds=None):
    """1号艇: 展示4位以下 かつ B級（不安2つ）、2号艇が展示2位以内 → 2号艇の単勝100円"""
    m = (df.er1 >= 4) & (df.gr1 <= 2) & (df.er2 <= 2)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "2"; x["amount"] = 100
    return x
