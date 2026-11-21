import pandas as pd, numpy as np
K = 2
def bets(df, odds=None):
    """1号艇のコース別1着率 cw1<35 かつ 2号艇モーター2連率 mo2>=37 のレース。
    3連単 2-1-X、X は 3〜6号艇のうち全国勝率 nw が高い順に上位K艇（同率は内側優先）。100円ずつ。"""
    m = (df.cw1 < 35) & (df.mo2 >= 37)
    s = df.loc[m].reset_index(drop=True)
    v = s[[f"nw{j}" for j in [3, 4, 5, 6]]].astype(float).fillna(-1).values
    rk = np.argsort(-v, axis=1, kind="stable")[:, :K] + 3
    out = []
    for i in range(K):
        x = s[["hd", "jcd", "rno"]].copy()
        x["ticket_type"] = "sanrensho"
        x["combination"] = ["2-1-%d" % t for t in rk[:, i]]
        x["amount"] = 100
        out.append(x)
    return pd.concat(out, ignore_index=True)
