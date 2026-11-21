import pandas as pd
# 2連単 2-1（2コース差し）。差しが出やすい場（探す期間で2-1の出現率>=7.0%）で、
# 1号艇の1コース1着率<=40% かつ 2号艇モーター2連率>=38%
SASHI_VENUES = [3, 4, 9, 10, 11, 14, 20, 22, 23]
def bets(df, odds=None):
    m = df.jcd.isin(SASHI_VENUES) & (df.cw1 <= 40) & (df.mo2 >= 38)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "nirensho"; x["combination"] = "2-1"; x["amount"] = 100
    return x
