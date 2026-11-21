import pandas as pd
def bets(df, odds=None):
    """雨（weather=='雨'、レース時記録＝近似）で、1号艇がB級（gr1<=2）、2号艇の展示タイムが1位のとき、単勝2を100円。"""
    m = (df.weather == "雨") & (df.gr1 <= 2) & (df.er2 == 1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "2"; r["amount"] = 100
    return r
