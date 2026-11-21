import pandas as pd
# 全場で、1号艇がB1・1コース1着率40%以上・全国勝率4.70以上なら単勝1を100円
def bets(df, odds=None):
    m = (df.gr1 == 2) & (df.cw1 >= 40) & (df.nw1 >= 4.7)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "1"; x["amount"] = 100
    return x
