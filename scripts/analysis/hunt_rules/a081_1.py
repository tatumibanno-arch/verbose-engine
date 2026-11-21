# a081_1: 前夜情報だけの LightGBM（艇ごとの1着確率, 2018-2020学習）→ Plackett-Luce で3連複20通りの確率。
# 最も確率が高い組が 1-2-3 で、その確率が 0.55 以上のレースだけ、3連複 1-2-3 を1点100円。
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/hunt/rules")
from a081_lib import long_feats, stack, trio_probs, TRIOS
M = pickle.load(open("/home/user/hunt/rules/a081_win_noex_2020.pkl", "rb"))
def bets(df, odds=None):
    df = df.reset_index(drop=True)
    X = stack(long_feats(df, False))
    p = M.predict_proba(X)[:, 1].reshape(6, -1).T
    tp = trio_probs(p); best = tp.argmax(1); pb = tp.max(1)
    m = (best == TRIOS.index((1, 2, 3))) & (pb >= 0.55)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "sanrenfuku"; r["combination"] = "1-2-3"; r["amount"] = 100
    return r
