import pandas as pd
def bets(df, odds=None):
    """今節の平均STが他5艇の平均より0.04以上遅い1号艇（B級）で、2号艇もB級のレース。単勝2を100円。
    （今節の前日までの成績なので初日は対象外）"""
    diff = df.ms1 - df[[f"ms{k}" for k in range(2, 7)]].mean(axis=1)
    m = (diff > 0.04) & (df.gr1 <= 2) & (df.gr2 <= 2)
    r = df.loc[m.fillna(False), ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "2"; r["amount"] = 100
    return r
