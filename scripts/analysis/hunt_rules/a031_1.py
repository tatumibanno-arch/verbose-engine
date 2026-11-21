import pandas as pd, numpy as np
WIND_MIN = 5      # 風速(m) 以上（レース時記録＝締切前表示の近似）
REL1_MAX = -2     # 1号艇の体重 - 6人平均 がこれ以下（1号艇が軽い）
GR1_MAX = 2        # 1号艇がB級（B1/B2）
def bets(df, odds=None):
    W = df[[f"wt{k}" for k in range(1, 7)]].astype(float)
    rel1 = df["wt1"].astype(float) - W.mean(axis=1)
    m = (df["wind"] >= WIND_MIN) & (rel1 <= REL1_MAX) & (df["gr1"] <= GR1_MAX)
    out = []
    for k in (2, 3, 4):
        x = df.loc[m & (df[f"gr{k}"] >= 3), ["hd", "jcd", "rno"]].copy()
        x["ticket_type"] = "tansho"; x["combination"] = str(k); x["amount"] = 100
        out.append(x)
    return pd.concat(out, ignore_index=True)
