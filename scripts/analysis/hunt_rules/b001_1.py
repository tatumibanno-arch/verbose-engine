import pandas as pd
def bets(df, odds=None):
    """締切時オッズで1号艇の市場シェア ph1>0.80 のレースで、1頭の3連単のうち締切時オッズが安い順に2点を100円ずつ"""
    r = df.loc[df.ph1 > 0.80, ["hd", "jcd", "rno"]]
    o = odds[odds.cb.str.startswith("1-")].merge(r, on=["hd", "jcd", "rno"])
    o = o.sort_values("odds").groupby(["hd", "jcd", "rno"]).head(2)
    out = o[["hd", "jcd", "rno"]].copy()
    out["ticket_type"] = "sanrensho"; out["combination"] = o.cb.values; out["amount"] = 100
    return out
