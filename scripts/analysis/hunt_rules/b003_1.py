import pandas as pd
def bets(df, odds=None):
    """1号艇が全国勝率で上位2位以内かつ展示タイム1位なのに、締切時オッズの1頭シェアが0.7未満のレースで、単勝1を100円。
    締切時オッズ・展示を使う（締切前アラート向け）。"""
    nw = df[[f"nw{k}" for k in range(1, 7)]]
    rk1 = nw.rank(axis=1, ascending=False, method="min")["nw1"]
    m = (rk1 <= 2) & (df.er1 == 1) & (df.ph1 < 0.7)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
