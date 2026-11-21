import pandas as pd
def bets(df, odds=None):
    """展示タイム1位が1号艇、展示2位・3位が2号艇と3号艇、2号艇のコース別2連率(ct2)>=50 のレースで 3連単 1-2-3 を1点"""
    m = (df.er1 == 1) & (df.er2 <= 3) & (df.er3 <= 3) & (df.ct2 >= 50)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "sanrensho"; r["combination"] = "1-2-3"; r["amount"] = 100
    return r
