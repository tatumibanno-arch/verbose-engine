import numpy as np, pandas as pd
def bets(df, odds=None):
    """1号艇が全国勝率6.0以上、2〜6号艇で全国勝率最上位の選手(全国6.0以上)が当地勝率で1.5以上低い
    （当地実績のない人気外枠選手を消す）レースで、単勝1を100円。"""
    NW = df[[f"nw{k}" for k in range(2, 7)]].values
    LW = df[[f"lw{k}" for k in range(2, 7)]].values
    i = np.nan_to_num(NW, nan=-1).argmax(1); idx = np.arange(len(df))
    bnw = NW[idx, i]; blw = LW[idx, i]
    m = (df.nw1.values >= 6.0) & (bnw >= 6.0) & (blw > 0) & (blw - bnw <= -1.5)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
