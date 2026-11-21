import pandas as pd
def bets(df, odds=None):
    """6コース2連率(ct6)が15%以上で2〜5号艇の各コース2連率以上、6コース出走数cn6>=5のレースで 2連単 1-6 を100円"""
    mx = df[["ct2", "ct3", "ct4", "ct5"]].max(axis=1)
    m = (df.ct6 >= 15) & (df.ct6 >= mx) & (df.cn6 >= 5)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "nirensho"; x["combination"] = "1-6"; x["amount"] = 100
    return x
