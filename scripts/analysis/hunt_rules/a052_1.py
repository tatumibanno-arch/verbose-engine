import pandas as pd
def bets(df, odds=None):
    """3連複 1-2-3 を1点。条件: 展示タイム1〜3号艇が上位3位を独占(er1..er3<=3)、
    1〜3号艇の全国勝率の最小 - 4〜6号艇の全国勝率の最大 > -1.83、風速<=3m（展示・風=締切前情報、風は近似）"""
    er = df[["er1", "er2", "er3"]].max(axis=1)
    gap = df[["nw1", "nw2", "nw3"]].min(axis=1) - df[["nw4", "nw5", "nw6"]].max(axis=1)
    m = (er <= 3) & (gap > -1.83) & (df.wind <= 3)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "sanrenfuku"; r["combination"] = "1-2-3"; r["amount"] = 100
    return r
