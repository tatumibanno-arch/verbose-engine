import pandas as pd
def bets(df, odds=None):
    """3号艇の3コース1着率>=30%、1号艇の1コース1着率<40% -> 単勝3を100円（前夜情報のみ）"""
    m = (df.cw3 >= 30) & (df.cw1 < 40)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "3"; r["amount"] = 100
    return r
