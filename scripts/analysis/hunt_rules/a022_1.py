import pandas as pd
def bets(df, odds=None):
    """2号艇のコース別平均STが3号艇より0.05以上遅く、3号艇の全国勝率が1号艇より1.0以上高いレースで、単勝3を100円"""
    m = ((df.cst2 - df.cst3) >= 0.05) & ((df.nw3 - df.nw1) >= 1.0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "3"; r["amount"] = 100
    return r
