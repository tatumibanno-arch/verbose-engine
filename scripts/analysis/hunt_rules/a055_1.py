import pandas as pd
def bets(df, odds=None):
    """12R で1号艇の展示タイムが1位、風速2m以下（風は近似）のレースで 3連単 1-全-全 20点"""
    m = (df.rno == 12) & (df.er1 == 1) & (df.wind <= 2)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for a in range(2, 7):
        for b in range(2, 7):
            if a == b: continue
            x = r.copy(); x["ticket_type"] = "sanrensho"; x["combination"] = f"1-{a}-{b}"; x["amount"] = 100; out.append(x)
    return pd.concat(out)
