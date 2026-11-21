import pandas as pd
def bets(df, odds=None):
    """1号艇のコース別1着率cw1>=40。2〜6号艇のうち、その枠の出走数cn<=9（欠損は経験ありとみなす）の艇を消す。
    残った艇が1〜2艇のレースで、2連単 1-残り艇 を各100円。"""
    m = df.cw1 >= 40
    keep = {k: ~(df[f"cn{k}"].fillna(99) <= 9) for k in range(2, 7)}
    nk = sum(keep[k].astype(int) for k in range(2, 7))
    m = m & nk.between(1, 2)
    out = []
    for k in range(2, 7):
        x = df.loc[m & keep[k], ["hd", "jcd", "rno"]].copy()
        x["ticket_type"] = "nirensho"; x["combination"] = f"1-{k}"; x["amount"] = 100
        out.append(x)
    return pd.concat(out)
