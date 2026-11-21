import pandas as pd
# a021_1: 期別の全体平均ST(ast)で、内隣(k-1号艇)が k号艇より0.07以上遅い（=k号艇がまくれる並び）、
# かつ 1号艇がA1でないレースで、k号艇(k=2,3,4)の単勝を100円。ast=0（欠損扱い）は除外。
def bets(df, odds=None):
    out = []
    for k in (2, 3, 4):
        m = ((df[f"ast{k-1}"] - df[f"ast{k}"]) >= 0.07) & (df[f"ast{k}"] >= 0.05) & (df["gr1"] <= 3)
        r = df.loc[m, ["hd", "jcd", "rno"]].copy()
        r["ticket_type"] = "tansho"; r["combination"] = str(k); r["amount"] = 100
        out.append(r)
    return pd.concat(out, ignore_index=True)
