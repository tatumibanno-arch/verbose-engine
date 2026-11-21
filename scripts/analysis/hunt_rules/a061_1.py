import pandas as pd
def bets(df, odds=None):
    """1号艇の選手が全国勝率4.5未満（格下）なのに、期別成績の1コース1着率が50%以上（出走8回以上）のとき、単勝1を100円"""
    m = (df.nw1 < 4.5) & (df.cw1 >= 50) & (df.cn1 >= 8)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
