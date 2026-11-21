import pandas as pd
def bets(df, odds=None):
    """1号艇: 展示4位以下 かつ 1コース平均ST0.18以上（不安2つ）、2号艇が展示1位 → 2号艇の単勝100円"""
    m = (df.er1 >= 4) & (df.cst1 >= 0.18) & (df.er2 == 1)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "2"; x["amount"] = 100
    return x
