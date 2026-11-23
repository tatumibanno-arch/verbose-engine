"""型の追加探索。G3=場ごと、G4=売れすぎ2艇を外す。発見（〜2026-03）／検証（2026-04〜）。締切時オッズ。"""
import sys, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); sys.path.insert(0,'scripts'); import overbought as ob; from common import VENUES
from sklearn.linear_model import LogisticRegression
D=ob.D; Rr=lambda s: round(s.pay.sum()/(100*len(s)),3) if len(s) else np.nan
def rep(nm,T,idc='i'): print(f'{nm} | 点 {len(T)} レース {T[idc].nunique()} 回収 {Rr(T)} 発見 {Rr(T[T.hd<"20260401"])} 検証 {Rr(T[T.hd>="20260401"])}')
C=pd.read_parquet(D+'ob3.parquet'); P=C[(C.gap>=.1)&(C.modiff<=-5)&(C.p1x>=.6)&(C.rer<C.xer)&(C.xer>=5)&(C.odds<15)]; A=P[P.p1x>=.7]
print('== G3 場ごと（候補A）'); g=A.groupby('jcd').apply(lambda s:pd.Series({'点':len(s),'全体':Rr(s),'発見':Rr(s[s.disc]),'検証':Rr(s[~s.disc])})); g.index=[VENUES[j] for j in g.index]; print(g.sort_values('全体',ascending=False).to_string())
gd=A[A.disc].groupby('jcd').apply(Rr); good=gd[gd>=1.0].index; bad=gd[gd<1.0].index
print('発見で1.0以上の場→検証',Rr(A[(~A.disc)&A.jcd.isin(good)]),len(A[(~A.disc)&A.jcd.isin(good)]),' 発見で1.0未満の場→検証',Rr(A[(~A.disc)&A.jcd.isin(bad)]),len(A[(~A.disc)&A.jcd.isin(bad)]))
print('場ごとの発見と検証の相関',round(g[['発見','検証']].corr().iloc[0,1],2))
# G4
R,L,oc=ob.load(); L=L.dropna(subset=['mo','nw']).copy()
X=lambda d:np.column_stack([d.r3,d.r2,d.mo/100,d.nw/10]+[(d.lane==k).astype(int) for k in range(3,7)])
disc=L[(L.hd<'20260401')&L.win1]; m=LogisticRegression(C=10,max_iter=500).fit(X(disc),disc.in3); L['f3']=m.predict_proba(X(L))[:,1]; L['gap']=L.mk3-L.f3
key=['hd','jcd','rno']; L['grk']=L.groupby(key).gap.rank(ascending=False,method='first'); Rk=R.set_index(key); rows=[]
for k,g in L[L.p1>=.6].groupby(key):
    if len(g)<5: continue
    g=g.sort_values('grk'); x1,x2=g.iloc[0],g.iloc[1]; rest=g.iloc[2:]
    if x2.gap<.05: continue
    o=Rk.loc[k]; ls=sorted(rest.lane)
    for a,b in itertools.permutations(ls,2): rows.append(k+(f'1-{a}-{b}',o[f'o{a}{b}'],o.res,o.payout,x1.gap,x2.gap,x1.mo-rest.mo.mean(),x2.mo-rest.mo.mean(),x1.er,x2.er,rest.er.mean(),x1.p1))
T=pd.DataFrame(rows,columns=key+['cb','odds','res','payout','g1','g2','m1','m2','e1','e2','re','p1']); T['pay']=np.where(T.cb==T.res,T.payout,0); T['i']=T.hd+T.jcd+T.rno.astype(str)
print('\n== G4 売れすぎ上位2艇（どちらも見立てより売れている）を外し、残り3艇で 1-abc-abc'); rep('全部',T)
for a in (15,30,60): rep(f'{a}倍未満',T[T.odds<a])
s=T[(T.m1<=-5)&(T.m2<=-5)]; rep('2艇ともモーター5以上低い',s); rep('　＋60倍未満',s[s.odds<60]); rep('　＋30倍未満',s[s.odds<30])
s=T[(T.e1>=4)&(T.e2>=4)]; rep('2艇とも展示4〜6位',s); rep('　＋60倍未満',s[s.odds<60]); rep('　＋30倍未満',s[s.odds<30]); rep('　＋30倍未満＆①頭70%以上',s[(s.odds<30)&(s.p1>=.7)])
s=T[(T.e1>=4)&(T.e2>=4)&(T.m1<=0)&(T.m2<=0)]; rep('2艇とも展示4〜6位＆モーターも残り以下',s); rep('　＋60倍未満',s[s.odds<60]); rep('　＋30倍未満',s[s.odds<30])
s=T[(T.re<T.e1)&(T.re<T.e2)]; rep('残りの展示が2艇とも上回る',s); rep('　＋30倍未満',s[s.odds<30]); rep('　＋60倍未満＆①頭70%以上',s[(s.odds<60)&(s.p1>=.7)])
