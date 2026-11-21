import pandas as pd
def bets(df, odds=None):
    """2号艇B級・6号艇A級・6号艇モーター2連率35%以上のレースで、3連単 1-6-2 を1点"""
    m = (df.gr2 <= 2) & (df.gr6 >= 3) & (df.mo6 >= 35)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "sanrensho"; r["combination"] = "1-6-2"; r["amount"] = 100
    return r
