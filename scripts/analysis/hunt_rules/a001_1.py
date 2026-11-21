import pandas as pd
def bets(df, odds=None):
    """1号艇A1かつ1コース1着率>=70%、2・3号艇の今節平均着順の平均>2.9（今節不調で人気が落ちている）のレースで3連単1-2-3を1点"""
    m23 = df[["mf2", "mf3"]].mean(axis=1, skipna=False)
    m = (df.gr1 == 4) & (df.cw1 >= 70) & (m23 > 2.9)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "sanrensho"; x["combination"] = "1-2-3"; x["amount"] = 100
    return x
