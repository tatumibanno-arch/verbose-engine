import numpy as np, pandas as pd
# 1号艇B級(全国勝率4.5未満)、5号艇か6号艇にA1(全国勝率6.5以上)。3連単 2-1-(A1の艇) を1点。
def bets(df, odds=None):
    out = []
    for A in (5, 6):
        m = (df.gr1 <= 2) & (df.nw1 < 4.5) & (df[f"gr{A}"] == 4) & (df[f"nw{A}"] >= 6.5)
        if A == 6:
            m &= df.gr5 != 4
        x = df.loc[m, ["hd", "jcd", "rno"]].copy()
        x["ticket_type"] = "sanrensho"; x["combination"] = f"2-1-{A}"; x["amount"] = 100
        out.append(x)
    return pd.concat(out)
