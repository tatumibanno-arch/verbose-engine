import pandas as pd
def bets(df, odds=None):
    """1号艇の展示順位2位以内・2号艇の展示順位5位以下・3号艇の展示順位3位以内のレースで、2連単 1-3 を100円"""
    m = (df.er1 <= 2) & (df.er2 >= 5) & (df.er3 <= 3)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "nirensho"; x["combination"] = "1-3"; x["amount"] = 100
    return x
