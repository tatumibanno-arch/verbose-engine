import pandas as pd
def bets(df, odds=None):
    """節3日目以降、1号艇の今節平均着順が他5艇の最良より1.0以上良く、1号艇の今節走数5以上 → 単勝1を100円"""
    oth = df[[f"mf{k}" for k in range(2, 7)]].min(axis=1)
    m = (df.day_no >= 3) & ((oth - df.mf1) >= 1.0) & (df.mn1 >= 5)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
