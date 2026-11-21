import pandas as pd
def bets(df, odds=None):
    """堅いレース指数: 1号艇の展示タイムが他5艇の最速より0.04秒超速い（締切前情報）、
    1号艇の1コース1着率>=50%、1号艇モーター2連率>=35%。単勝1を100円。"""
    etmin = df[[f"et{k}" for k in range(2, 7)]].min(axis=1)
    m = ((etmin - df.et1) > 0.04) & (df.cw1 >= 50) & (df.mo1 >= 35)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
