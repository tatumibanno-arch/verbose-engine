import pandas as pd
def bets(df, odds=None):
    """1号艇のコース1着率(cw1)が20%未満、2号艇がB1級、2号艇のモーター2連率35%以上 -> 単勝2を100円（前夜情報のみ）"""
    m = (df.cw1 < 20) & (df.gr2 == 2) & (df.mo2 >= 35)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "2"; r["amount"] = 100
    return r
