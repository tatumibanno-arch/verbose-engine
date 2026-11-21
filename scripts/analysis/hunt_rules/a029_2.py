import sys, pandas as pd
sys.path.insert(0, "/home/user/hunt/rules")
from a029_1 import _base
def bets(df, odds=None):
    """混合戦で1号艇が女子、展示1位が2〜4号艇の男子 → 2連単 その男子-（1号艇と自分以外）の4点"""
    r = _base(df)
    out = []
    for j in range(2, 7):
        x = r[r.fk != j].copy()
        x["ticket_type"] = "nirensho"; x["combination"] = x.fk.astype(str) + "-" + str(j); x["amount"] = 100
        out.append(x)
    return pd.concat(out)[["hd", "jcd", "rno", "ticket_type", "combination", "amount"]]
