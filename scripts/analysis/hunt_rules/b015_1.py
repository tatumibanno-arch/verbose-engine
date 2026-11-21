# b015_1: 機械学習の1着確率（オッズ以外・展示含む、2020-01〜2025-02で学習）p と締切時3連単から出した市場シェア ph を比べ、
# 1号艇の p>=0.45 かつ p/ph>=1.25 のレースで単勝1を100円
import sys; sys.path.insert(0, "/home/user/hunt/rules")
import pandas as pd, b015_feat as f
def bets(df, odds=None):
    P = f.winprob(df); p = P[:, 0]; r = p / df.ph1.values
    m = (p >= 0.45) & (r >= 1.25)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy(); x["ticket_type"] = "tansho"; x["combination"] = "1"; x["amount"] = 100
    return x
