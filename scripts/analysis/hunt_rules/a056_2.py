import pandas as pd
def bets(df, odds=None):
    """6号艇の能力指数58以上でスタートが遅め(ast6>0.16)。3連単 1-2-6, 1-3-6 を各100円"""
    m = (df.abil6 >= 58) & (df.ast6 > 0.16)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for cb in ["1-2-6", "1-3-6"]:
        x = r.copy(); x["ticket_type"] = "sanrensho"; x["combination"] = cb; x["amount"] = 100; out.append(x)
    return pd.concat(out)
