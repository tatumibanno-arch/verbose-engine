import numpy as np, pandas as pd
def bets(df, odds=None):
    """強風（wind>=5m、レース時記録で締切前表示の近似）で1号艇がB級のとき、
    3・4号艇のうち全国勝率の高い方（1号艇より勝率が上、モーター2連率36%超）の単勝を100円。"""
    b3 = df.nw3 >= df.nw4
    k = np.where(b3, 3, 4)
    nw = np.where(b3, df.nw3, df.nw4)
    mo = np.where(b3, df.mo3, df.mo4)
    m = (df.wind >= 5) & (df.gr1 <= 2) & (nw > df.nw1) & (mo > 36)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"
    r["combination"] = pd.Series(k, index=df.index)[m].astype(str).values
    r["amount"] = 100
    return r
