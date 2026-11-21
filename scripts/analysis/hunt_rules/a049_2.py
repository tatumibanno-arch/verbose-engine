import pandas as pd
def bets(df, odds=None):
    """2連単 2-1 を1点。条件: 1号艇の1コース1着率<=40%、2号艇の全国勝率が1号艇より0.5以上高い。前夜に分かる情報だけ。"""
    m = (df.cw1 <= 40) & ((df.nw2 - df.nw1) >= 0.5)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "nirensho"; x["combination"] = "2-1"; x["amount"] = 100
    return x
