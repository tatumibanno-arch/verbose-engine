import pandas as pd
def bets(df, odds=None):
    """5・6号艇で、今節モーター2連対率(mm)>=0.6(今節6走以上)なのに番組表モーター2連率(mo)<30 の艇がいれば、ワイド 1-その艇 を100円"""
    out = []
    for k in (5, 6):
        m = (df[f"mmn{k}"] >= 6) & (df[f"mm{k}"] >= 0.6) & (df[f"mo{k}"] < 30)
        x = df.loc[m, ["hd", "jcd", "rno"]].copy()
        x["ticket_type"] = "wide"; x["combination"] = f"1-{k}"; x["amount"] = 100
        out.append(x)
    return pd.concat(out, ignore_index=True)
