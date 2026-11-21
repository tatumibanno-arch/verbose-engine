import pandas as pd
# a006_1 に「1コース平均STが0.19以下」を足したもの(単勝1を100円)
WEAK_IN = {1, 2, 3, 4, 14}
def bets(df, odds=None):
    m = (~df.jcd.isin(WEAK_IN)) & (df.gr1 <= 2) & (df.cw1 >= 50) & (df.cst1 <= 0.19)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
