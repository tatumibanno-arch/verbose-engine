"""a080_1: 単純なロジスティック回帰（6クラス、1着艇）で「1号艇が1着にならない確率」pf を出す。
学習は探す期間 A_disc の 2016-01〜2019-12 だけ（固定）。特徴量は前夜情報＋展示タイム/展示順位（締切前情報）。
pf>=0.80 のレースで、2〜6号艇のうち予測1着確率の上位2艇を頭、2・3着は上位3艇（1号艇は入れない）の3連単 4点。"""
import pandas as pd, numpy as np, itertools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
BASE=['et','er','nw','n2','lw','l2','mo','bo','gr','cw','ct','cst','ast','abil','wt','ms','mf','mm']
THR=0.80
def _feats(d):
    parts={}
    for b in BASE:
        M=d[[f'{b}{k}' for k in range(1,7)]].astype(float)
        mu=M.mean(axis=1)
        for k in range(1,7):
            parts[f'{b}{k}']=M[f'{b}{k}'].fillna(mu)
            parts[f'{b}{k}_d']=(M[f'{b}{k}']-mu).fillna(0)
    X=pd.DataFrame(parts,index=d.index)
    for j in range(1,25): X[f'j{j}']=(d.jcd.astype(int)==j).astype(float)
    return X
_M=None
def _model():
    global _M
    if _M is None:
        cols=['hd','w1']+[f'{b}{k}' for b in BASE for k in range(1,7)]+['jcd']
        tr=pd.read_parquet('/home/user/hunt/A_disc_races.parquet',columns=cols)
        tr=tr[tr.hd<'20200101'].dropna(subset=['w1'])
        X=_feats(tr); med=X.median(); X=X.fillna(med)
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,C=0.1))
        m.fit(X,tr.w1.astype(int)); _M=(m,med)
    return _M
def bets(df, odds=None):
    m,med=_model()
    X=_feats(df).fillna(med)
    P=m.predict_proba(X)
    pf=1-P[:,0]
    order=np.argsort(-P[:,1:],axis=1)+2
    out=[]
    keys=df[['hd','jcd','rno']].values
    for i in np.where(pf>=THR)[0]:
        top3=list(order[i,:3])
        for h in top3[:2]:
            for a,b in itertools.permutations([x for x in top3 if x!=h],2):
                out.append((keys[i][0],keys[i][1],keys[i][2],'sanrensho',f'{h}-{a}-{b}',100))
    return pd.DataFrame(out,columns=['hd','jcd','rno','ticket_type','combination','amount'])
