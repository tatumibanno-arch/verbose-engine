import pandas as pd, numpy as np
def bets(df, odds=None):
    """3連複1点: 各艇の点数 = その枠のコース別2連率 ct + 30*(展示タイムの平均からの速さ/0.07)。
    上位3艇の3連複を、3位と4位の点数差が20以上のレースだけ買う（展示=締切前情報）"""
    L = range(1, 7)
    et = df[[f"et{k}" for k in L]].to_numpy(dtype=float)
    ct = df[[f"ct{k}" for k in L]].fillna(0).to_numpy(dtype=float)
    etz = np.nan_to_num(-(et - np.nanmean(et, axis=1, keepdims=True)) / 0.07)
    S = ct + 30 * etz
    o = np.argsort(-S, axis=1)
    ss = -np.sort(-S, axis=1)
    m = (ss[:, 2] - ss[:, 3]) >= 20
    top = np.sort(o[:, :3] + 1, axis=1)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "sanrenfuku"
    r["combination"] = ["-".join(map(str, t)) for t in top[m]]
    r["amount"] = 100
    return r
