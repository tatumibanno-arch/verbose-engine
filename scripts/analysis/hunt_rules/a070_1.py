import numpy as np, pandas as pd
def bets(df, odds=None):
    """人気になりそうな艇=全国勝率最上位のA1が2〜6号艇にいて、その艇の展示順位が4位以下（不安材料）→消し。
    1号艇の展示順位が1位なら1号艇の単勝を100円。展示を使う（締切前情報）。"""
    nw = df[[f"nw{k}" for k in range(1, 7)]].values
    fav = np.nanargmax(np.nan_to_num(nw, nan=-1), axis=1)
    idx = np.arange(len(df))
    fgr = df[[f"gr{k}" for k in range(1, 7)]].values[idx, fav]
    fer = df[[f"er{k}" for k in range(1, 7)]].values[idx, fav]
    m = (fav >= 1) & (fgr == 4) & (fer >= 4) & (df.er1.values == 1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
