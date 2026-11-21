import pandas as pd
def bets(df, odds=None):
    """1号艇のコース1着率cw1>=70、2号艇がB級(gr2<=2)でコース2連率ct2>=50。nirenfuku 1-2 を100円"""
    m = (df.cw1 >= 70) & (df.ct2 >= 50) & (df.gr2 <= 2)
    r = df.loc[m, ["hd", "jcd", "rno"]].copy()
    r["ticket_type"] = "nirenfuku"; r["combination"] = "1-2"; r["amount"] = 100
    return r
