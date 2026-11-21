import numpy as np, pandas as pd
def bets(df, odds=None):
    """荒れ指数: 1号艇の全国勝率が他5艇の最高より2.5以上低いレース（1号艇が弱い＝荒れやすい）。
    1号艇以外でその枠のコース別1着率(cw)が最も高い艇を単勝100円。ただしその艇がA1でない（級<=A2）とき。"""
    nw = df[[f"nw{k}" for k in range(1, 7)]].values
    gap1 = df.nw1.values - np.nanmax(nw[:, 1:], axis=1)
    cw = np.nan_to_num(df[[f"cw{k}" for k in range(1, 7)]].values, nan=-1)
    top = np.argmax(cw, axis=1) + 1
    gr = df[[f"gr{k}" for k in range(1, 7)]].values[np.arange(len(df)), top - 1]
    m = (gap1 <= -2.5) & (top != 1) & (gr <= 3)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = top[m].astype(str); r["amount"] = 100
    return r
