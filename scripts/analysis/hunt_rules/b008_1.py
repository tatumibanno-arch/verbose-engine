import pandas as pd
def bets(df, odds=None):
    """締切時オッズで「B級なのに売れている」艇: B級(gr<=2)・2〜6号艇・3連単由来の1着シェアph>0.20、
    かつ1号艇のシェアph1<0.35（イン不安の混戦）。その艇の単勝を100円。"""
    out = []
    for k in range(2, 7):
        m = (df[f"gr{k}"] <= 2) & (df[f"ph{k}"] > 0.20) & (df["ph1"] < 0.35)
        x = df.loc[m, ["hd", "jcd", "rno"]].copy()
        x["ticket_type"] = "tansho"; x["combination"] = str(k); x["amount"] = 100
        out.append(x)
    return pd.concat(out, ignore_index=True)
