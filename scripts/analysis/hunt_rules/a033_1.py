import pandas as pd
# 探す期間の前半(2016-2019)で、風速4m以上のとき1号艇の1着率が場の平均以上だった(場,風向)の組 = 「イン有利な風」(近似)
TAIL = {(2, 5), (2, 9), (3, 1), (3, 3), (4, 9), (6, 7), (7, 5), (7, 15), (8, 7), (10, 9), (13, 9), (16, 3), (16, 7), (18, 7),
        (21, 1), (21, 3), (22, 1), (22, 3), (22, 15), (23, 9), (23, 13), (23, 15), (24, 1), (24, 9)}
def bets(df, odds=None):
    """イン有利な風向(場ごとに学習)で風速3m以上、1号艇がB級 → 単勝1を100円（風は締切前表示の近似）"""
    key = pd.Series(list(zip(df.jcd.astype(int), df.wdir.fillna(-1).astype(int))), index=df.index)
    m = key.isin(TAIL) & (df.wind >= 3) & (df.gr1 <= 2)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
