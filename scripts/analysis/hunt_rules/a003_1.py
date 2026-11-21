import pandas as pd
def bets(df, odds=None):
    """1号艇A1・2〜6号艇が全員B級のレースで、6号艇がB1かつモーター2連率37%以上なら 2連単 1-6 を100円"""
    m = (df.gr1 == 4) & (df[[f"gr{k}" for k in range(2, 7)]].max(axis=1) <= 2)
    m &= (df.gr6 == 2) & (df.mo6 >= 37)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "nirensho"; x["combination"] = "1-6"; x["amount"] = 100
    return x
