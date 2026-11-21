import pandas as pd
def bets(df, odds=None):
    """進入固定レース（race_name に「進入固定」）で、1号艇がA2かB1のとき、単勝1を100円。
    テーマ: レース番号ごとの傾向を調べた結果、場ごとの1R〜8Rに多い進入固定戦のインだけが0.97前後と高かった。"""
    m = df.race_name.astype(str).str.contains("進入固定") & df.gr1.isin([2, 3])
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
