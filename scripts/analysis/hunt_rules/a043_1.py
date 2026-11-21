import pandas as pd
def bets(df, odds=None):
    """企画レース（1号艇だけA級、2〜6号艇は全員B級）で、1号艇の全国勝率が他5艇の最高より1.5以上高く、
    1号艇のモーター2連率35%以上のとき、単勝1を100円。前夜に分かる情報だけ。"""
    oth_gr = df[[f"gr{k}" for k in range(2, 7)]].max(axis=1)
    oth_nw = df[[f"nw{k}" for k in range(2, 7)]].max(axis=1)
    m = (df.gr1 >= 3) & (oth_gr <= 2) & ((df.nw1 - oth_nw) >= 1.5) & (df.mo1 >= 35)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
