import pandas as pd
def bets(df, odds=None):
    """同じ条件（6号艇 全国勝率6.0以上・モーター45%以上、1号艇 勝率6.0未満）で ワイド 1-6 と 2-6 を100円ずつ"""
    m = (df.nw6 >= 6.0) & (df.mo6 >= 45) & (df.nw1 < 6.0)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for cb in ["1-6", "2-6"]:
        x = r.copy(); x["ticket_type"] = "wide"; x["combination"] = cb; x["amount"] = 100; out.append(x)
    return pd.concat(out)
