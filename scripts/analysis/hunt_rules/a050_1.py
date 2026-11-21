"""a050_1: 2連複の期待値モデル（締切前情報＝展示タイム・展示順位を使う）
探す期間(2016-2022)全部で学習した2つのLightGBM（的中確率 / 的中時の配当の対数）で
15通りの各組の期待値 EV=p*E[配当] を出し、EV>1.1 の組を100円ずつ買う。
モデルファイル: a050_m1.txt, a050_m2.txt（smearing係数は a050_smear.txt）"""
import pandas as pd, numpy as np, itertools, os, lightgbm as lgb
D=os.path.dirname(os.path.abspath(__file__))
BF=['gr','nw','n2','lw','l2','mo','bo','cw','ct','cst','abil','et','er','mf','mm','ms']
TH=1.1
def build(d):
    d=d.reset_index(drop=True); n=len(d); out=[]
    B={f:d[[f'{f}{k}' for k in range(1,7)]].to_numpy(float) for f in BF}
    rk=(-B['nw']).argsort(1).argsort(1)+1
    for i,j in itertools.combinations(range(1,7),2):
        X={'ridx':np.arange(n),'i':np.full(n,i),'j':np.full(n,j)}
        for f,A in B.items():
            X[f+'_i']=A[:,i-1]; X[f+'_j']=A[:,j-1]
        X['rk_i']=rk[:,i-1]; X['rk_j']=rk[:,j-1]
        oth=[k for k in range(6) if k not in (i-1,j-1)]
        X['oth_nwmax']=B['nw'][:,oth].max(1); X['oth_cwmax']=B['cw'][:,oth].max(1)
        X['nw1']=B['nw'][:,0]; X['cw1']=B['cw'][:,0]
        X['oth_ermin']=B['er'][:,oth].min(1); X['er1']=B['er'][:,0]
        out.append(pd.DataFrame(X))
    L=pd.concat(out,ignore_index=True)
    L['jcd']=d.jcd.to_numpy()[L.ridx]; L['rno']=d.rno.to_numpy()[L.ridx]
    L['hd']=d.hd.to_numpy()[L.ridx]
    L['combination']=L.i.astype(str)+'-'+L.j.astype(str)
    return L
FEATS=None
def bets(df, odds=None):
    L=build(df)
    m1=lgb.Booster(model_file=os.path.join(D,'a050_m1.txt')); m2=lgb.Booster(model_file=os.path.join(D,'a050_m2.txt'))
    smear=float(open(os.path.join(D,'a050_smear.txt')).read())
    F=m1.feature_name()
    X=L[F].copy(); X['jcd']=X.jcd.astype(int)
    ev=m1.predict(X)*np.exp(m2.predict(X))*smear/100
    r=L.loc[ev>TH,['hd','jcd','rno','combination']].copy()
    r['ticket_type']='nirenfuku'; r['amount']=100
    return r[['hd','jcd','rno','ticket_type','combination','amount']]
