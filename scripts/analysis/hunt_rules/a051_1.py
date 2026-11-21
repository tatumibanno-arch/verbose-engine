import pandas as pd
def bets(df, odds=None):
    """6号艇: 全国勝率6.0以上かつモーター2連率45%以上、1号艇: 全国勝率6.0未満 → ワイド 1-6 を100円"""
    m = (df.nw6 >= 6.0) & (df.mo6 >= 45) & (df.nw1 < 6.0)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "wide"; x["combination"] = "1-6"; x["amount"] = 100
    return x
