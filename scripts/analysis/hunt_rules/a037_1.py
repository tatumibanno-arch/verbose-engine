import pandas as pd
def bets(df, odds=None):
    """締切19時前（ナイター帯を除く）のレースで、1号艇がB1かつ全国勝率が他5艇の最高-1.0以上 → 単勝1を100円"""
    h = df.close_time.str[:2].astype(int)
    gap = df.nw1 - df[[f"nw{k}" for k in range(2, 7)]].max(axis=1)
    m = (h < 19) & (df.gr1 == 2) & (gap >= -1.0)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
