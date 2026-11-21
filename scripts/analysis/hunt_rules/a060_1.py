import pandas as pd
def bets(df, odds=None):
    """内枠(1-3)の能力指数の最大 <= 外枠(4-6)の最小、外枠最大と内枠最大の差>=8、
    1号艇の1コース1着率<45% のレースで、ワイド 4-5,4-6,5-6 を各100円"""
    ia = df[["abil1", "abil2", "abil3"]].max(axis=1)
    oa = df[["abil4", "abil5", "abil6"]]
    m = (oa.min(axis=1) >= ia) & (oa.max(axis=1) - ia >= 8) & (df.cw1 < 45)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for cb in ["4-5", "4-6", "5-6"]:
        x = r.copy(); x["ticket_type"] = "wide"; x["combination"] = cb; x["amount"] = 100; out.append(x)
    return pd.concat(out)
