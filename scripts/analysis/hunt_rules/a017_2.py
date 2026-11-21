import pandas as pd
def bets(df, odds=None):
    """逆向きの食い違い: 1号艇の番組表モーター2連率(mo)>=40 なのに今節2連対率(mm)<0.3(今節6走以上)のレースで、単勝1を100円"""
    m = (df["mmn1"] >= 6) & (df["mm1"] < 0.3) & (df["mo1"] >= 40)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "1"; x["amount"] = 100
    return x
