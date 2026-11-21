import pandas as pd
def bets(df, odds=None):
    """外枠(4-6)の全国勝率の最大 < 5.0 かつ 内枠(1-3)の全国勝率の最小 - 外枠の最大 >= 1.0 のレースで、3連単 1-2-3,1-3-2,2-1-3 を100円ずつ"""
    omax = df[["nw4", "nw5", "nw6"]].max(axis=1)
    imin = df[["nw1", "nw2", "nw3"]].min(axis=1)
    m = (omax < 5.0) & ((imin - omax) >= 1.0)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for cb in ["1-2-3", "1-3-2", "2-1-3"]:
        x = r.copy(); x["ticket_type"] = "sanrensho"; x["combination"] = cb; x["amount"] = 100; out.append(x)
    return pd.concat(out)
