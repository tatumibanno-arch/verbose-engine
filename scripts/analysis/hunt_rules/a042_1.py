import pandas as pd
def bets(df, odds=None):
    """節の初日(day_no==1)、1号艇の1コース1着率(期別)が39〜55%で能力指数48以下（中堅の1号艇）のレースで、単勝1を100円。前夜情報のみ。"""
    m = (df.day_no == 1) & (df.cw1 >= 39) & (df.cw1 <= 55) & (df.abil1 <= 48)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
