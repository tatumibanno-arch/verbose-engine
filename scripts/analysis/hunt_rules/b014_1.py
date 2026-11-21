import pandas as pd
def bets(df, odds=None):
    """締切時の単勝シェア(ph)が0.15以上、期のモーター2連率>=40なのに今節のモーター2連対率<=0.34(今節3走以上)の艇の単勝を100円"""
    out = []
    for k in range(1, 7):
        m = (df[f"ph{k}"] >= 0.15) & (df[f"mo{k}"] >= 40) & (df[f"mm{k}"] <= 0.34) & (df[f"mmn{k}"] >= 3)
        x = df.loc[m, ["hd", "jcd", "rno"]].copy()
        x["ticket_type"] = "tansho"; x["combination"] = str(k); x["amount"] = 100
        out.append(x)
    return pd.concat(out, ignore_index=True)
