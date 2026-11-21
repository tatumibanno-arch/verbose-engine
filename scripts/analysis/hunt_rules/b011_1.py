import pandas as pd
def bets(df, odds=None):
    """締切時オッズで 3連単 1-2-3 が 6.5倍以下のレースだけ、1-2-3 を100円（表目の本命が低オッズのとき）。"""
    o = odds[odds.cb == "1-2-3"][["hd", "jcd", "rno", "odds"]]
    r = df[["hd", "jcd", "rno"]].merge(o, on=["hd", "jcd", "rno"])
    r = r[r.odds <= 6.5][["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "sanrensho"; r["combination"] = "1-2-3"; r["amount"] = 100
    return r
