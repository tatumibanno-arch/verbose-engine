import pandas as pd, numpy as np
def bets(df, odds=None):
    """3連単 i-j-全(4点,各100円)。頭i=2〜5号艇、展示順位 i<=2位・j<=3位、
    締切時オッズの合成オッズ(1/Σ1/odds)が15〜30倍。展示=締切前情報、オッズは確定オッズ(締切前とは違う)。"""
    o = odds[odds.odds > 0].copy()
    o["ij"] = o.cb.str[:3]
    g = o.groupby(["hd", "jcd", "rno", "ij"]).odds.agg(lambda s: 1 / (1 / s).sum() if len(s) == 4 else np.nan).rename("syn").reset_index()
    g = g[(g.syn >= 15) & (g.syn <= 30)]
    g["i"] = g.ij.str[0].astype(int); g["j"] = g.ij.str[2].astype(int)
    g = g[g.i.between(2, 5)]
    d = df[["hd", "jcd", "rno"] + [f"er{k}" for k in range(1, 7)]]
    g = g.merge(d, on=["hd", "jcd", "rno"])
    er = g[[f"er{k}" for k in range(1, 7)]].to_numpy()
    idx = np.arange(len(g))
    g = g[(er[idx, g.i.to_numpy() - 1] <= 2) & (er[idx, g.j.to_numpy() - 1] <= 3)]
    out = []
    for k in range(1, 7):
        x = g[(g.i != k) & (g.j != k)][["hd", "jcd", "rno", "ij"]].copy()
        x["combination"] = x.ij + f"-{k}"
        out.append(x)
    r = pd.concat(out).drop(columns="ij")
    r["ticket_type"] = "sanrensho"; r["amount"] = 100
    return r[["hd", "jcd", "rno", "ticket_type", "combination", "amount"]]
