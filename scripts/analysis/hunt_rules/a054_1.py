import pandas as pd
def bets(df, odds=None):
    """1号艇の1コース平均ST(期別)が0.22より遅いレースで、3連単 2-{3,4,5,6}-1 の4点を100円ずつ（1号艇3着型）"""
    m = df.cst1 > 0.22
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for b in [3, 4, 5, 6]:
        x = r.copy(); x["ticket_type"] = "sanrensho"; x["combination"] = f"2-{b}-1"; x["amount"] = 100; out.append(x)
    return pd.concat(out)
