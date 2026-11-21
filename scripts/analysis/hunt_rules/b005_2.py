import pandas as pd
# 1番人気オッズ<=6倍、かつ1号艇の頭シェア ph1<=0.6 のレースで、人気1〜3位の3連単を100円ずつ
def bets(df, odds=None):
    o = odds[odds.odds > 0].sort_values(["hd", "jcd", "rno", "odds"])
    o["rk"] = o.groupby(["hd", "jcd", "rno"]).cumcount() + 1
    top = o[o.rk <= 3]
    fav = top[top.rk == 1]
    ok = fav[fav.odds <= 6.0][["hd", "jcd", "rno"]].merge(df.loc[df.ph1 <= 0.6, ["hd", "jcd", "rno"]], on=["hd", "jcd", "rno"])
    x = top.merge(ok, on=["hd", "jcd", "rno"])[["hd", "jcd", "rno", "cb"]].rename(columns={"cb": "combination"})
    x["ticket_type"] = "sanrensho"; x["amount"] = 100
    return x[["hd", "jcd", "rno", "ticket_type", "combination", "amount"]]
