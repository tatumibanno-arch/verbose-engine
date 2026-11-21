import pandas as pd
def bets(df, odds=None):
    """5号艇の全国勝率が2・3号艇より高く、5号艇モーター2連率40%以上のレースで、3連単 1-5-2 を1点"""
    m = (df.nw5 > df.nw2) & (df.nw5 > df.nw3) & (df.mo5 >= 40)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "sanrensho"; r["combination"] = "1-5-2"; r["amount"] = 100
    return r
