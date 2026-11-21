import pandas as pd
def bets(df, odds=None):
    """2連単 3-2 を1点。条件: 2号艇の全国勝率>=1号艇、風速>=4m（レース時記録＝締切前表示の近似）、3号艇の展示順位<=3。締切前情報を使う。"""
    m = (df.nw2 >= df.nw1) & (df.wind >= 4) & (df.er3 <= 3)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "nirensho"; x["combination"] = "3-2"; x["amount"] = 100
    return x
