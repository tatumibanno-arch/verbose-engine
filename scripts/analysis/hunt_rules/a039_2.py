import pandas as pd
def bets(df, odds=None):
    """進入固定レース（レース名に「進入固定」）で、1号艇が40歳以下なら単勝1を100円"""
    rn = df.race_name.astype(str).str.replace(r'[\s　]', '', regex=True)
    m = rn.str.contains('進入固定') & (df.age1 <= 40)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "tansho"; r["combination"] = "1"; r["amount"] = 100
    return r
