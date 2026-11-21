import pandas as pd
def bets(df, odds=None):
    """例: 1号艇がA1で他が全員B級のレースで、3連単 1-2-3 と 1-3-2 を100円ずつ"""
    m = (df.gr1 == 4) & (df[[f"gr{k}" for k in range(2, 7)]].max(axis=1) <= 2)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for cb in ["1-2-3", "1-3-2"]:
        x = r.copy(); x["ticket_type"] = "sanrensho"; x["combination"] = cb; x["amount"] = 100; out.append(x)
    return pd.concat(out)
