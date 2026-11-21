import pandas as pd
def bets(df, odds=None):
    """冬（12・1・2月）に、2号艇のコース別1着率(cw2)が6艇のコース別1着率の中で最大のレースで、単勝2を100円。"""
    m_ = df.hd.astype(str).str[4:6].astype(int)
    cw = df[[f"cw{k}" for k in range(1, 7)]].astype(float)
    top2 = cw.rank(axis=1, ascending=False, method="min")["cw2"] == 1
    m = m_.isin([12, 1, 2]) & top2
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "2"; x["amount"] = 100
    return x
