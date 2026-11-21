import pandas as pd
# 全場版: 1号艇の1コース1着率<=40% かつ 2号艇モーター2連率>=38% で 2連単 2-1
def bets(df, odds=None):
    m = (df.cw1 <= 40) & (df.mo2 >= 38)
    x = df.loc[m, ["hd", "jcd", "rno"]].copy()
    x["ticket_type"] = "nirensho"; x["combination"] = "2-1"; x["amount"] = 100
    return x
