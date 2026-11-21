import pandas as pd
def bets(df, odds=None):
    """5コース1着率が高く(cw5>=15, 出走数cn5>=18)、1号艇の1コース1着率が低い(cw1<40)レースで単勝5を100円"""
    m = (df.cw5 >= 15) & (df.cn5 >= 18) & (df.cw1 < 40)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "5"; r["amount"] = 100
    return r
