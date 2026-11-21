import numpy as np, pandas as pd
MO1_MAX = 25      # 1号艇のモーター2連率がこれ以下（極端に低い）
CMO_MIN = 40      # 2〜4号艇でモーター2連率が最も高い艇のモーター2連率がこれ以上
GAP_MIN = 1.0     # その艇の全国勝率が1号艇より GAP_MIN 以上高い
def bets(df, odds=None):
    """1号艇のモーターが極端に弱いレースで1号艇を消し、2〜4号艇のうちモーター最上位の艇の単勝を100円"""
    mo = df[["mo2", "mo3", "mo4"]].values.astype(float)
    nw = df[["nw2", "nw3", "nw4"]].values.astype(float)
    mo_f = np.where(np.isnan(mo), -1, mo)
    k = mo_f.argmax(1); i = np.arange(len(df))
    cmo = mo[i, k]; cnw = nw[i, k]
    m = (df.mo1.values <= MO1_MAX) & (cmo >= CMO_MIN) & (cnw - df.nw1.values >= GAP_MIN)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = (k[m] + 2).astype(str); r["amount"] = 100
    return r
