import numpy as np, pandas as pd, itertools
# 全国勝率の上位3艇が1・2・3号艇、勝率差(上位3平均−下位3平均)2.0以上、1号艇A1 → 3連単 1,2,3 ボックス6点
def bets(df, odds=None):
    S = df[[f"nw{k}" for k in range(1, 7)]].fillna(0).values
    ss = -np.sort(-S, axis=1)
    gap = ss[:, :3].mean(1) - ss[:, 3:].mean(1)
    top = np.sort(np.argsort(-S, axis=1)[:, :3] + 1, axis=1)
    m = (top[:, 0] == 1) & (top[:, 1] == 2) & (top[:, 2] == 3) & (gap >= 2.0) & (df.gr1.values == 4)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for p in itertools.permutations([1, 2, 3]):
        x = r.copy(); x["ticket_type"] = "sanrensho"; x["combination"] = "-".join(map(str, p)); x["amount"] = 100; out.append(x)
    return pd.concat(out)
