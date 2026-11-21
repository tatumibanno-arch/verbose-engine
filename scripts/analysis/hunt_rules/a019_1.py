import pandas as pd
def bets(df, odds=None):
    """4号艇の今節平均STが速く(<=0.13, 今節3走以上)、1〜3号艇の最速より0.05以上速いとき、4号艇の単勝を100円"""
    gap = df[["ms1", "ms2", "ms3"]].min(axis=1) - df.ms4
    m = (df.ms4 <= 0.13) & (df.mn4 >= 3) & (gap >= 0.05)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "4"; r["amount"] = 100
    return r
