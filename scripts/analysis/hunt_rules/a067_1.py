import pandas as pd, numpy as np
def _rank(df, p):
    X = df[[f"{p}{k}" for k in range(1, 7)]].values.astype(float)
    # 順位 = 自分より大きい艇の数 + 1（同値は同順位）
    return (X[:, :, None] < X[:, None, :]).sum(2) + 1
def bets(df, odds=None):
    """1号艇がB1・全国勝率がレース内4位以内・ボート2連率順位+モーター2連率順位<=4 → 単勝1を100円"""
    br = _rank(df, "bo"); mr = _rank(df, "mo"); nr = _rank(df, "nw")
    m = (df.gr1.values == 2) & (nr[:, 0] <= 4) & (br[:, 0] + mr[:, 0] <= 4)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
