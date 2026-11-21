import pandas as pd
def bets(df, odds=None):
    """進入固定レース（レース名に「進入固定」）で、1号艇が36歳以下かつモーター2連率30%以上なら単勝1を100円"""
    rn = df.race_name.astype(str).str.replace(r'[\s　]', '', regex=True)
    m = rn.str.contains('進入固定') & (df.age1 <= 36) & (df.mo1 >= 30)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
