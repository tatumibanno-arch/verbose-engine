import pandas as pd
def bets(df, odds=None):
    """締切時オッズで1号艇頭シェアが50〜65%のレースで、2連単 3-1 を100円"""
    m = (df.ph1 >= 0.50) & (df.ph1 < 0.65)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirensho"; r["combination"] = "3-1"; r["amount"] = 100
    return r
