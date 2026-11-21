import pandas as pd
# 平日（土日・主な固定祝日/お盆/正月を除く）の1〜4R、1号艇の1コース1着率(cw1)が40%超60%以下 → 単勝1 を100円
HOL_MD = {101,102,103,111,211,223,320,321,429,503,504,505,720,811,813,814,815,923,1103,1123}
def bets(df, odds=None):
    d = pd.to_datetime(df.hd.astype(str), format="%Y%m%d")
    md = d.dt.month * 100 + d.dt.day
    hol = (d.dt.dayofweek >= 5) | md.isin(HOL_MD)
    m = (~hol) & (df.rno <= 4) & (df.cw1 > 40) & (df.cw1 <= 60)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "tansho"; x["combination"] = "1"; x["amount"] = 100
    return x
