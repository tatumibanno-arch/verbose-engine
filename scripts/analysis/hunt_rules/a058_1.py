import pandas as pd
def bets(df, odds=None):
    """2・3号艇の全国勝率がともに5.0未満、1号艇と5号艇の全国勝率がともに6.0以上のレースで、2連複 1-5 を100円"""
    m = (df.nw2 < 5.0) & (df.nw3 < 5.0) & (df.nw1 >= 6.0) & (df.nw5 >= 6.0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirenfuku"; r["combination"] = "1-5"; r["amount"] = 100
    return r
