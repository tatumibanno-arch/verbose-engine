import pandas as pd
def bets(df, odds=None):
    """6号艇がA1、1号艇の頭シェア(ph1)<=0.45 のレースで、
    3連単「X-Y-6」(Xは1以外、6が3着)のうち締切時オッズ200倍以下の目を全部100円ずつ。"""
    r = df.loc[(df.gr6 == 4) & (df.ph1 <= 0.45), ["hd", "jcd", "rno"]]
    o = odds.merge(r, on=["hd", "jcd", "rno"])
    o = o[(o.cb.str[4] == "6") & (o.cb.str[0] != "1") & (o.odds <= 200)]
    out = o[["hd", "jcd", "rno"]].copy()
    out["ticket_type"] = "sanrensho"; out["combination"] = o.cb.values; out["amount"] = 100
    return out
