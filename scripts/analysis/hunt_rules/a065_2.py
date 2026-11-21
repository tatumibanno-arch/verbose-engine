import pandas as pd
def bets(df, odds=None):
    """5号艇がB級(gr5<=2)だが5コース1着率が高く(cw5>=13)、1号艇の1コース1着率が低い(cw1<45)レースで単勝5を100円"""
    m = (df.cw5 >= 13) & (df.gr5 <= 2) & (df.cw1 < 45)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "5"; r["amount"] = 100
    return r
