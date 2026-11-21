import numpy as np, pandas as pd
def bets(df, odds=None):
    """展示タイムを「場・日ごとの全艇平均との差」に直し、1号艇がレース内で最も速く（差<-0.03秒、2番手との差>=0.03秒）、1号艇がB1級なら単勝1を100円。"""
    etc = [f"et{k}" for k in range(1, 7)]
    et = df[etc].astype(float).values.copy()
    et[(et < 6.0) | (et > 7.5)] = np.nan
    lg = pd.DataFrame(et); lg["hd"] = df.hd.values; lg["jcd"] = df.jcd.values
    dm = lg.melt(["hd", "jcd"]).groupby(["hd", "jcd"])["value"].mean().rename("dm").reset_index()
    dmv = df[["hd", "jcd"]].merge(dm, on=["hd", "jcd"], how="left")["dm"].values
    z = et - dmv[:, None]
    zf = np.where(np.isnan(z), 9, z)
    zs = np.sort(zf, axis=1)
    best = np.argmin(zf, axis=1) + 1
    m = (best == 1) & (zs[:, 0] < -0.03) & ((zs[:, 1] - zs[:, 0]) >= 0.03) & (df.gr1.values == 2)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
