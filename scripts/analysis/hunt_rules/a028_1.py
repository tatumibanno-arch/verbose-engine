import numpy as np, pandas as pd
# 全国勝率の上位3艇が1・2・3号艇で、上位3艇平均−下位3艇平均の全国勝率差が2.5以上 → 3連複 1-2-3 を1点
def bets(df, odds=None):
    S = df[[f"nw{k}" for k in range(1, 7)]].fillna(0).values
    ss = -np.sort(-S, axis=1)
    gap = ss[:, :3].mean(1) - ss[:, 3:].mean(1)
    top = np.sort(np.argsort(-S, axis=1)[:, :3] + 1, axis=1)
    m = (top[:, 0] == 1) & (top[:, 1] == 2) & (top[:, 2] == 3) & (gap >= 2.5)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "sanrenfuku"; r["combination"] = "1-2-3"; r["amount"] = 100
    return r
