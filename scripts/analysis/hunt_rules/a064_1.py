import pandas as pd
def bets(df, odds=None):
    """2号艇のコース別1着率が1号艇より5pt以上低くない(cw2>=cw1-5)のに、全国勝率は1号艇より低い(nw2<nw1)レース。単勝2 と 2連単2-1 を100円ずつ"""
    m = (df.cw2 >= df.cw1 - 5) & (df.nw2 < df.nw1)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for tt, cb in [("tansho", "2"), ("nirensho", "2-1")]:
        x = r.copy(); x["ticket_type"] = tt; x["combination"] = cb; x["amount"] = 100; out.append(x)
    return pd.concat(out)
