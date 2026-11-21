import pandas as pd
def bets(df, odds=None):
    """鳴門(jcd=14)だけ: 3号艇がA級(gr3>=3)のレースで2連単 3-2 を100円。
    場ごとの配当傾向（鳴門は3コース頭・2着2号艇の目が他場より払い戻しが良い）を使う。"""
    m = (df.jcd == 14) & (df.gr3 >= 3)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirensho"; r["combination"] = "3-2"; r["amount"] = 100
    return r
