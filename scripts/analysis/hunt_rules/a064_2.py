import pandas as pd
def bets(df, odds=None):
    """2号艇のコース別1着率が1号艇以上(cw2>=cw1)で、全国勝率は1号艇より低い(nw2<nw1)レース。2連単2-1 を100円"""
    m = (df.cw2 >= df.cw1) & (df.nw2 < df.nw1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirensho"; r["combination"] = "2-1"; r["amount"] = 100
    return r
