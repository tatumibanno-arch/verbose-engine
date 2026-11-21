"""a079_1: LightGBM（A_disc の 2016-2019 で学習、前夜情報のみ）で各艇の1着確率を出し、
1号艇がB級（gr1<=2）かつ 1号艇の確率>0.5 のレースで単勝1を100円。"""
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
BF=['gr','nw','n2','lw','l2','mo','bo','age','wt','sex','cw','ct','cst','cn','ast','abil','ms','mf','mn','mm','mmn']
def _long(df):
    parts=[]
    for k in range(1,7):
        x=pd.DataFrame({f:df[f'{f}{k}'].astype(float).values for f in BF}); x['lane']=k; x['ridx']=np.arange(len(df)); parts.append(x)
    L=pd.concat(parts,ignore_index=True)
    for f in ['nw','n2','abil','mo','cw','ct']:
        g=L.groupby('ridx')[f]; L[f+'_d']=L[f]-g.transform('mean'); L[f+'_r']=g.rank(ascending=False)
    L['jcd']=np.tile(df.jcd.astype(int).values,6); L['dist']=np.tile(pd.to_numeric(df.distance,errors='coerce').values,6)
    return L
def bets(df, odds=None):
    df=df.reset_index(drop=True)
    b=lgb.Booster(model_file=str(Path(__file__).with_name('a079_model.txt')))
    L=_long(df); F=b.feature_name()
    P=b.predict(L[F].values.astype(np.float32),num_threads=2).reshape(6,-1).T; P=P/P.sum(1,keepdims=True)
    m=(df.gr1<=2).values&(P[:,0]>0.5)
    r=df.loc[m,['hd','jcd','rno']].copy(); r['ticket_type']='tansho'; r['combination']='1'; r['amount']=100
    return r
