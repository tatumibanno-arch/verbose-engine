import pandas as pd
# イン(1号艇)1着率が低い場(探す期間で50%未満: 桐生・戸田・江戸川・平和島・鳴門)を除き、
# 1号艇がB級(B1/B2)なのに、その選手の1コース1着率(期別)が50%以上のレースで、単勝1を100円。
WEAK_IN = {1, 2, 3, 4, 14}
def bets(df, odds=None):
    m = (~df.jcd.isin(WEAK_IN)) & (df.gr1 <= 2) & (df.cw1 >= 50)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
