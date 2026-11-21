import numpy as np, pandas as pd
def _base(df):
    S = df[[f"sex{k}" for k in range(1, 7)]].values
    nf = (S == 2).sum(1)
    ER = df[[f"er{k}" for k in range(1, 7)]].values
    has = (ER == 1).any(1)
    fk = np.where(has, (ER == 1).argmax(1) + 1, 0)
    fsex = np.where(fk > 0, S[np.arange(len(df)), np.clip(fk - 1, 0, 5)], 0)
    m = (nf >= 1) & (nf <= 5) & (df.sex1.values == 2) & (fk >= 2) & (fk <= 4) & (fsex == 1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["fk"] = fk[m]
    return r
def bets(df, odds=None):
    """混合戦で1号艇が女子、展示タイム1位が2〜4号艇の男子 → その男子の単勝100円"""
    r = _base(df)
    r["ticket_type"] = "tansho"; r["combination"] = r.fk.astype(str); r["amount"] = 100
    return r[["hd", "jcd", "rno", "ticket_type", "combination", "amount"]]
