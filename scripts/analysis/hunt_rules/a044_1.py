import pandas as pd
def bets(df, odds=None):
    """距離1200m（荒天短縮）のレースで、展示タイム（締切前情報）を使う単勝。
    4号艇の展示順位が2位以内なら4号艇の単勝、1号艇の展示順位が1位なら1号艇の単勝。各100円。"""
    s = df[df.distance == 1200]
    out = []
    for lane, m in [(4, s.er4 <= 2), (1, s.er1 == 1)]:
        x = s.loc[m, ["hd", "jcd", "rno"]].copy()
        x["ticket_type"] = "tansho"; x["combination"] = str(lane); x["amount"] = 100
        out.append(x)
    return pd.concat(out, ignore_index=True)
