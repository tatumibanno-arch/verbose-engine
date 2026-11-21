"""a050_2: 2連複 1-2 を100円。条件: 2号艇の展示順位が2位以内 かつ 2号艇の全国勝率が4.3〜5.7（中堅）。
締切前情報（展示）を使う。"""
import pandas as pd
def bets(df, odds=None):
    m=(df.er2<=2)&(df.nw2>=4.3)&(df.nw2<=5.7)
    r=df.loc[m,['hd','jcd','rno']].copy()
    r['ticket_type']='nirenfuku'; r['combination']='1-2'; r['amount']=100
    return r
