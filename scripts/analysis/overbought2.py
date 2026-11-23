"""overbought.py の続き: 売れすぎの相手Xを外す6点に、展示・節間モーター・節間成績を足すとどこまで上がるか。
注意: オッズは締切時（締切前とは違う）。発見=〜2026-03、検証=2026-04〜。"""
import sqlite3, sys, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
from sklearn.linear_model import LogisticRegression
c=sqlite3.connect('data/boatrace.db')
R,L,oc=ob.load(); L=L.dropna(subset=['mo','nw']).copy()
X=lambda d:np.column_stack([d.r3,d.r2,d.mo/100,d.nw/10]+[(d.lane==k).astype(int) for k in range(3,7)])
disc=L[(L.hd<'20260401')&L.win1]; m=LogisticRegression(C=10,max_iter=500).fit(X(disc),disc.in3); L['f3']=m.predict_proba(X(L))[:,1]; L['gap']=L.mk3-L.f3
# 節間（前日まで）のモーター・選手の平均着順
e=pd.read_sql("select hd,jcd,rno,lane,racer_no,motor_no from entries where hd>='20250320'",c)
r=pd.read_sql("select hd,jcd,rno,lane,position from results where hd>='20250320'",c).merge(e,on=['hd','jcd','rno','lane'])
r['pos']=pd.to_numeric(r.position,errors='coerce').fillna(6)
dd=r[['jcd','hd']].drop_duplicates().sort_values(['jcd','hd']); dd['d']=pd.to_datetime(dd.hd); dd['meet']=(dd.groupby('jcd').d.diff().dt.days!=1).groupby(dd.jcd).cumsum()
r=r.merge(dd[['jcd','hd','meet']],on=['jcd','hd'])
def prior(key,nm):
    g=r.groupby(['jcd','meet',key,'hd']).pos.agg(['sum','count']).reset_index().sort_values('hd')
    g['cs']=g.groupby(['jcd','meet',key])['sum'].cumsum()-g['sum']; g['cc']=g.groupby(['jcd','meet',key])['count'].cumsum()-g['count']
    g[nm]=np.where(g.cc>0,g.cs/g.cc,np.nan); g[nm+'n']=g.cc; return g[['jcd','hd',key,nm,nm+'n']]
e=e.merge(prior('motor_no','mm'),on=['jcd','hd','motor_no'],how='left').merge(prior('racer_no','rm'),on=['jcd','hd','racer_no'],how='left')
b=pd.read_sql("select hd,jcd,rno,lane,start_st st from beforeinfo where hd>='20250401'",c); b['st']=pd.to_numeric(b.st,errors='coerce'); b['sr']=b.groupby(['hd','jcd','rno']).st.rank(method='min')
L=L.merge(e[['hd','jcd','rno','lane','mm','mmn','rm']],on=['hd','jcd','rno','lane'],how='left').merge(b[['hd','jcd','rno','lane','sr']],on=['hd','jcd','rno','lane'],how='left')
E1=pd.read_parquet(ob.D+'ob_ent.parquet'); e1=E1[E1.lane==1][['hd','jcd','rno','er','mo']].rename(columns={'er':'er1','mo':'mo1'})
L['grk']=L.groupby(['hd','jcd','rno']).gap.rank(ascending=False,method='first')
rows=[]
for (hd,j,rn),g in L[L.p1>=.4].groupby(['hd','jcd','rno']):
    if len(g)<5: continue
    x=g[g.grk==1].iloc[0]; rest=g[g.grk!=1].sort_values('f3',ascending=False).head(3)
    rows.append((hd,j,rn,x.lane,x.gap,x.mo-rest.mo.mean(),x.er,rest.er.mean(),x.mm,rest.mm.mean(),x.rm,rest.rm.mean(),x.sr,rest.sr.mean(),x.p1,''.join(map(str,sorted(rest.lane)))))
B=pd.DataFrame(rows,columns=['hd','jcd','rno','X','gap','modiff','xer','rer','xmm','rmm','xrm','rrm','xsr','rsr','p1x','ls']).merge(R,on=['hd','jcd','rno']).merge(e1,on=['hd','jcd','rno'],how='left')
B['inv6']=[sum(1/r[f'o{a}{b}'] for a,b in itertools.permutations(r['ls'],2)) for _,r in B.iterrows()]
B['syn']=1/B.inv6; B['mk6']=B.inv6/B.tot; B['hit']=[r.res in [f"1-{a}-{b}" for a,b in itertools.permutations(r.ls,2)] for r in B.itertuples()]; B['pay']=np.where(B.hit,B.payout,0)
B.to_parquet(ob.D+'ob2_bets.parquet')
def t(nm,s):
    o=[nm,'| n',len(s),'的中',round(s.hit.mean(),3),'売れ方',round(s.mk6.mean(),3),'合成',round(s.syn.median(),1),'回収',round(s.pay.sum()/(600*len(s)),3)]
    for h,g in (('発見',s[s.hd<'20260401']),('検証',s[s.hd>='20260401'])): o+=[h,len(g),round(g.pay.sum()/(600*max(len(g),1)),3)]
    print(*o)
base=B[(B.gap>=.1)&(B.modiff<=-5)&(B.p1x>=.6)]
print('--- 事前に決めた区切り（土台: 売れすぎ幅0.1以上・Xのモーター5以上低い・①頭60%以上）')
t('土台',base)
t('＋Xの展示4〜6位',base[base.xer>=4]); t('＋Xの展示1〜2位',base[base.xer<=2])
t('＋残り3艇の展示平均がXより良い',base[base.rer<base.xer])
t('＋①の展示1〜2位',base[base.er1<=2]); t('＋①の展示5〜6位',base[base.er1>=5])
t('＋Xの節間モーター着順が残りより悪い',base[base.xmm>base.rmm]); t('＋Xの節間モーター着順が残りより1以上悪い',base[base.xmm>base.rmm+1])
t('＋Xの節間の本人着順が残りより悪い',base[base.xrm>base.rrm])
t('＋Xのスタ展4〜6番手',base[base.xsr>=4])
t('＋全部（X展示4-6・節間モーター悪い・①展示1-2）',base[(base.xer>=4)&(base.xmm>base.rmm)&(base.er1<=2)])
# モデル: 発見で学習し、検証で「見立て÷売れ方」の高い順に区切る
F=B[B.p1x>=.4].dropna(subset=['xer','rer','er1','xsr','rsr']).copy()
for cc in ('xmm','rmm','xrm','rrm'): F[cc]=F[cc].fillna(3.5)
feat=lambda d:np.column_stack([np.log(d.mk6/(1-d.mk6)),d.gap,d.modiff/10,d.xer,d.rer,d.er1,d.xmm-d.rmm,d.xrm-d.rrm,d.xsr,d.rsr,d.mo1/100,(d.X==2),(d.X==3)])
tr=F[F.hd<'20260401']; va=F[F.hd>='20260401'].copy(); mdl=LogisticRegression(C=1,max_iter=2000).fit(feat(tr),tr.hit)
print('\n係数(logit売れ方,gap,modiff,X展示,残り展示,①展示,節間モ差,節間本人差,Xスタ展,残りスタ展,①モ,X=2,X=3):',mdl.coef_[0].round(3))
va['pred']=mdl.predict_proba(feat(va))[:,1]; va['edge']=va.pred/va.mk6
print('\n--- 検証期間（2026-04〜11）: モデルの「見立て÷売れ方」が高い順')
for lo,hi in ((0,.9),(.9,1),(1,1.05),(1.05,1.1),(1.1,1.2),(1.2,9)):
    s=va[(va.edge>=lo)&(va.edge<hi)]
    if len(s): print(f'{lo}〜{hi}','n',len(s),'的中',round(s.hit.mean(),3),'売れ方',round(s.mk6.mean(),3),'合成',round(s.syn.median(),1),'回収',round(s.pay.sum()/(600*len(s)),3))
top=va[va.edge>=1.1]; mo=top.groupby(top.hd.str[:6]).apply(lambda s:pd.Series({'n':len(s),'roi':round(s.pay.sum()/(600*len(s)),2)})); print(mo.T.to_string())
