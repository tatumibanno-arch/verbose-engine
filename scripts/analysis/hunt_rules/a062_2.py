import pandas as pd
def bets(df, odds=None):
    """6コース2連率(ct6)が20%以上かつ6コース出走数cn6>=10のレースで 2連単 1-6 を100円（広めの版）"""
    m = (df.ct6 >= 20) & (df.cn6 >= 10)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "nirensho"; x["combination"] = "1-6"; x["amount"] = 100
    return x
