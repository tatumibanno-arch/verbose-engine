import pandas as pd
def bets(df, odds=None):
    """若手の1号艇（28歳以下）が年上の相手（2〜6号艇の平均年齢40歳以上）と走り、全国勝率が相手の最高より1.0以上は劣らないレースで、単勝1を100円"""
    others_age = df[[f"age{k}" for k in range(2, 7)]].mean(axis=1)
    others_nw = df[[f"nw{k}" for k in range(2, 7)]].max(axis=1)
    m = (df.age1 <= 28) & (others_age >= 40) & (df.nw1 - others_nw > -1.0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
