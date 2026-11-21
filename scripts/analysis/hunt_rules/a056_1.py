import pandas as pd
def bets(df, odds=None):
    """6号艇が実力上位(abil6>54)だがスタートが遅め(ast6>0.16)、1号艇の1コース1着率55%以上。3連単 1-2-6, 1-3-6 を各100円"""
    m = (df.abil6 > 54) & (df.ast6 > 0.16) & (df.cw1 >= 55)
    r = df.loc[m, ["hd", "jcd", "rno"]]
    out = []
    for cb in ["1-2-6", "1-3-6"]:
        x = r.copy(); x["ticket_type"] = "sanrensho"; x["combination"] = cb; x["amount"] = 100; out.append(x)
    return pd.concat(out)
