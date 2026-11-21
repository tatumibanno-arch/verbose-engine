import pandas as pd
def bets(df, odds=None):
    """A1が4〜6号艇にだけいるレースで、1号艇がB1・1コース1着率(cw1)>=46%・モーター2連率(mo1)>=40%なら単勝1を100円"""
    inner = df[["gr1", "gr2", "gr3"]].eq(4).any(axis=1)
    outer = df[["gr4", "gr5", "gr6"]].eq(4).any(axis=1)
    m = ~inner & outer & (df.gr1 == 2) & (df.cw1 >= 46) & (df.mo1 >= 40)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
