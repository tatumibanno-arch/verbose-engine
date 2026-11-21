import pandas as pd
def bets(df, odds=None):
    """節3日目以降、1号艇の全国勝率が6艇中3位以内かつモーター2連率が6艇中1位 → 単勝1を100円"""
    nwr = df[[f"nw{k}" for k in range(1, 7)]].rank(axis=1, ascending=False, method="min")
    mor = df[[f"mo{k}" for k in range(1, 7)]].rank(axis=1, ascending=False, method="min")
    m = (df.day_no >= 3) & (nwr["nw1"] <= 3) & (mor["mo1"] <= 1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
