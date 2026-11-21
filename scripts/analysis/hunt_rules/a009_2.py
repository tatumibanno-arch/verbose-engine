import pandas as pd
# 4号艇の全体平均STが1〜3号艇の最速より0.02以上早い ＋ 風速4m以上（近似）＋ 4号艇の展示タイム順位2位以内 → 単勝4 を100円
def bets(df, odds=None):
    ok = (df[["ast1", "ast2", "ast3", "ast4"]] > 0).all(axis=1)
    s = (df[["ast1", "ast2", "ast3"]].min(axis=1) - df.ast4).round(2)
    m = ok & (s >= 0.02) & (df.wind >= 4) & (df.er4 <= 2)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "4"; r["amount"] = 100
    return r
