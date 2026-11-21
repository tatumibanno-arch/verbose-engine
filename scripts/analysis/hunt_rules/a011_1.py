import pandas as pd
def bets(df, odds=None):
    """1号艇がB級(B1/B2)で、展示タイムが他の5艇の誰よりも0.045秒超遅い（展示単独最下位）レース。
    1号艇を消して 単勝 2 と 3 を100円ずつ。展示タイムは締切前情報。"""
    others = df[[f"et{k}" for k in range(2, 7)]].max(axis=1)
    m = (df.gr1 <= 2) & ((df.et1 - others) > 0.045)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for cb in ["2", "3"]:
        x = r.copy(); x["ticket_type"] = "tansho"; x["combination"] = cb; x["amount"] = 100; out.append(x)
    return pd.concat(out)
