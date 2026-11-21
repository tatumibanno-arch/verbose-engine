import pandas as pd
# 1コースが強い9場で、1号艇がB1・1コース1着率40%以上・全国勝率4.70以上なら単勝1を100円
TOP = [3, 5, 8, 13, 16, 17, 18, 19, 24]
def bets(df, odds=None):
    m = df.jcd.isin(TOP) & (df.gr1 == 2) & (df.cw1 >= 40) & (df.nw1 >= 4.7)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "1"; x["amount"] = 100
    return x
