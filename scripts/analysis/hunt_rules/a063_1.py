import pandas as pd
def bets(df, odds=None):
    """3号艇の3コース1着率>=20%、1号艇の1コース1着率<50%、3号艇が展示タイム1位 -> 単勝3を100円（展示=締切前情報を使用）"""
    m = (df.cw3 >= 20) & (df.cw1 < 50) & (df.er3 == 1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "3"; r["amount"] = 100
    return r
