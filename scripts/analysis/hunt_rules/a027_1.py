import itertools
import numpy as np
import pandas as pd

# 混戦（ct/10+全国勝率 の上位1位と4位の差 <= 3.76）のレースで、
# 強さ = ct/10 + nw - 60*(展示タイム - レース平均) の上位3艇をワイドBOX（3点）
SPREAD_MAX = 3.76
ET_W = 60.0
KIND = "wide"

def _score(df):
    ct = np.nan_to_num(df[[f"ct{k}" for k in range(1, 7)]].values.astype(float))
    nw = np.nan_to_num(df[[f"nw{k}" for k in range(1, 7)]].values.astype(float))
    base = ct / 10 + nw
    et = df[[f"et{k}" for k in range(1, 7)]].values.astype(float)
    etz = np.nan_to_num(et - np.nanmean(et, axis=1, keepdims=True))
    return base, base - ET_W * etz

def bets(df, odds=None):
    df = df.reset_index(drop=True)
    base, s2 = _score(df)
    srt = -np.sort(-base, axis=1)
    m = (srt[:, 0] - srt[:, 3]) <= SPREAD_MAX
    order = np.argsort(-s2, axis=1)[:, :3] + 1
    out = []
    sub = df.loc[m, ["hd", "jcd", "rno"]].reset_index(drop=True)
    top = order[m]
    size = 2 if KIND == "wide" else 3
    for c in itertools.combinations(range(3), size):
        x = sub.copy()
        boats = np.sort(top[:, list(c)], axis=1)
        x["ticket_type"] = KIND
        x["combination"] = ["-".join(map(str, r)) for r in boats]
        x["amount"] = 100
        out.append(x)
    return pd.concat(out, ignore_index=True)
