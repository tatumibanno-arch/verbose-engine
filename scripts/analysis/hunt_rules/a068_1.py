import numpy as np
import pandas as pd

def bets(df, odds=None):
    """そろい踏み1号艇・格下: 1号艇が モーター2連率が場内2位以内・展示タイム1位・今節平均着順が2位以内(今節2走以上の艇の中)
    で、かつ全国勝率が他艇最高より1.0以上低い → 単勝 1 を100円。展示(締切前情報)を使う。"""
    mo = df[[f"mo{k}" for k in range(1, 7)]].values
    morank1 = (mo[:, 1:] > mo[:, [0]]).sum(1) + 1
    mf = df[[f"mf{k}" for k in range(1, 7)]].where(df[[f"mn{k}" for k in range(1, 7)]].values >= 2)
    mfr1 = mf.rank(axis=1, method="min").iloc[:, 0].values
    nwgap = df.nw1.values - df[[f"nw{k}" for k in range(2, 7)]].max(axis=1).values
    m = (morank1 <= 2) & (df.er1.values == 1) & (mfr1 <= 2) & (nwgap <= -1.0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
