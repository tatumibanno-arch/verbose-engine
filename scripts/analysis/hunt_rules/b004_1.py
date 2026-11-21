import pandas as pd
def bets(df, odds=None):
    """b004_1: 展示タイム1位が1号艇のレースで、締切時3連単オッズ6倍未満の目をすべて100円ずつ（締切時オッズ使用＝締切前より甘い）"""
    r = df.loc[df.er1 == 1, ["hd", "jcd", "rno"]]
    o = odds[odds.odds < 6].merge(r, on=["hd", "jcd", "rno"])
    out = o[["hd", "jcd", "rno"]].copy()
    out["ticket_type"] = "sanrensho"; out["combination"] = o.cb.values; out["amount"] = 100
    return out
