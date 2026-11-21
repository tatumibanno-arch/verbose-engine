import pandas as pd
# 締切時オッズ1番人気の3連単の頭が1号艇以外、かつ1番人気オッズ<=8倍のレースで、人気1〜3位の3連単を100円ずつ
MAX_FAV = 8.0
def bets(df, odds=None):
    o = odds[odds.odds > 0].sort_values(["hd", "jcd", "rno", "odds"])
    o["rk"] = o.groupby(["hd", "jcd", "rno"]).cumcount() + 1
    top = o[o.rk <= 3]
    fav = top[top.rk == 1]
    ok = fav[(fav.cb.str[0] != "1") & (fav.odds <= MAX_FAV)][["hd", "jcd", "rno"]]
    ok = ok.merge(df[["hd", "jcd", "rno"]], on=["hd", "jcd", "rno"])
    x = top.merge(ok, on=["hd", "jcd", "rno"])[["hd", "jcd", "rno", "cb"]].rename(columns={"cb": "combination"})
    x["ticket_type"] = "sanrensho"; x["amount"] = 100
    return x[["hd", "jcd", "rno", "ticket_type", "combination", "amount"]]
