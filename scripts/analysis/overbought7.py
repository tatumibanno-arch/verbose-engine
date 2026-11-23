"""型の追加探索。G2=2連単で売れすぎを外す、G1=①以外が頭で堅いレースで同じことをする。締切時オッズ（締切前とは違う）。"""
import sqlite3, sys, json, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
from sklearn.linear_model import LogisticRegression
c=sqlite3.connect('data/boatrace.db'); D=ob.D; key=['hd','jcd','rno']
Rr=lambda s: round(s.pay.sum()/(100*len(s)),3) if len(s) else np.nan
def rep(nm,T,cut): print(f'{nm} | 点 {len(T)} レース {T.i.nunique()} 回収 {Rr(T)} 前半 {Rr(T[T.hd<cut])} 後半 {Rr(T[T.hd>=cut])}')
E=pd.read_parquet(D+'ob_ent.parquet'); 
for q in ('mo','nw'): E[q]=pd.to_numeric(E[q],errors='coerce')
E['cn']=E.cn.fillna(0)
# ---------- G2 2連単
o=pd.read_sql("select hd,jcd,rno,combination cb,odds from odds where ticket_type='nirensho' and odds>0",c); o['inv']=1/o.odds
tot=o.groupby(key).inv.agg(tot='sum',n='size'); h=o[o.cb.str[0]=='1'].copy(); h['lane']=h.cb.str[2].astype(int); h=h.merge(tot.reset_index(),on=key); h=h[h.n>=28]
h['h1']=h.groupby(key).inv.transform('sum'); h['p1']=h.h1/h.tot; h['mk2']=h.inv/h.h1
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='nirensho' and hd>='20251101'",c).drop_duplicates(key)
h=h.merge(p,on=key).merge(E,on=key+['lane'],how='left').dropna(subset=['mo','nw','er'])
h['win1']=h.res.str[0]=='1'; h['in2']=h.res.str[2]==h.lane.astype(str); b2=h[h.win1].groupby('lane').in2.mean(); h['r2']=(h.c2.fillna(0)+10*h.lane.map(b2))/(h.cn+10)
X=lambda d:np.column_stack([d.r2,d.mo/100,d.nw/10]+[(d.lane==k).astype(int) for k in range(3,7)])
cut='20260501'; tr=h[(h.hd<cut)&h.win1]; m=LogisticRegression(C=10,max_iter=500).fit(X(tr),tr.in2); h['f2']=m.predict_proba(X(h))[:,1]; h['gap']=h.mk2-h.f2
h['grk']=h.groupby(key).gap.rank(ascending=False,method='first'); h['i']=h.hd+h.jcd+h.rno.astype(str)
x=h[h.grk==1][key+['gap','mo','er','lane']].rename(columns={'gap':'xgap','mo':'xmo','er':'xer','lane':'X'})
r=h[h.grk!=1].merge(x,on=key); r['rmo']=r.groupby(key).mo.transform('mean'); r['rer']=r.groupby(key).er.transform('mean'); r['pay']=np.where(r.in2&r.win1,r.payout,0)
print('== G2 2連単: ①頭で、2着の売れすぎXを外して残り4艇へ（期間',h.hd.min(),'〜',h.hd.max(),'）'); rep('全部',r[r.p1>=.6],cut)
b=r[(r.p1>=.6)&(r.xgap>=.08)&(r.xmo-r.rmo<=-5)]; rep('売れすぎ幅0.08以上＆Xのモーター5以上低い',b,cut)
for a in (6,10,15): rep(f'　＋{a}倍未満',b[b.odds<a],cut)
b2_=b[(b.xer>=5)&(b.rer<b.xer)]; rep('　＋X展示5〜6位',b2_,cut)
for a in (6,10,15): rep(f'　　＋{a}倍未満',b2_[b2_.odds<a],cut)
rep('　　＋10倍未満＆①頭70%以上',b2_[(b2_.odds<10)&(b2_.p1>=.7)],cut); rep('　　＋15倍未満＆①頭70%以上',b2_[(b2_.odds<15)&(b2_.p1>=.7)],cut); rep('　　＋15倍未満＆①頭80%以上',b2_[(b2_.odds<15)&(b2_.p1>=.8)],cut)
rep('　　＋相手の展示が1〜2位の組だけ＆15倍未満',b2_[(b2_.er<=2)&(b2_.odds<15)],cut)
# ---------- G1 ①以外が頭で堅い
rows=[]
for mth in pd.period_range('2025-04','2026-11',freq='M'):
    a=mth.strftime('%Y%m01'); bb=mth.strftime('%Y%m31')
    o=pd.read_sql(f"select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd between '{a}' and '{bb}' and odds>0",c)
    if o.empty: continue
    o['inv']=1/o.odds; o['h']=o.cb.str[0].astype(int); t=o.groupby(key).inv.agg(tot='sum',n='size').reset_index(); hs=o.groupby(key+['h']).inv.sum().rename('hs').reset_index().merge(t,on=key); hs['ph']=hs.hs/hs.tot
    f=hs[(hs.n>=110)&(hs.h!=1)&(hs.ph>=.45)][key+['h','ph','hs']]; rows.append(o.merge(f,on=key+['h']))
O=pd.concat(rows); p3=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250401'",c).drop_duplicates(key); O=O.merge(p3,on=key)
O['a']=O.cb.str[2].astype(int); O['b']=O.cb.str[4].astype(int); O['i']=O.hd+O.jcd+O.rno.astype(str); print('\n== G1 ①以外の頭が45%以上売れているレース',O.i.nunique())
ln=[]
for k in range(1,7):
    s=O[(O.a==k)|(O.b==k)].groupby(key+['h','ph','hs','res']).inv.sum().rename('s').reset_index(); s['lane']=k; ln.append(s)
Ln=pd.concat(ln); Ln=Ln[Ln.lane!=Ln.h]; Ln['mk3']=Ln.s/Ln.hs; Ln=Ln.merge(E,on=key+['lane'],how='left').dropna(subset=['mo','nw','er'])
Ln['hwin']=Ln.res.str[0]==Ln.h.astype(str); Ln['in3']=[str(l) in r[2:] for l,r in zip(Ln.lane,Ln.res)]; b3=Ln[Ln.hwin].groupby('lane').in3.mean(); Ln['r3']=(Ln.c3.fillna(0)+10*Ln.lane.map(b3))/(Ln.cn+10)
X3=lambda d:np.column_stack([d.r3,d.mo/100,d.nw/10]+[(d.lane==k).astype(int) for k in range(2,7)])
cut='20260401'; tr=Ln[(Ln.hd<cut)&Ln.hwin]; m3=LogisticRegression(C=10,max_iter=500).fit(X3(tr),tr.in3); Ln['f3']=m3.predict_proba(X3(Ln))[:,1]; Ln['gap']=Ln.mk3-Ln.f3
Ln['grk']=Ln.groupby(key).gap.rank(ascending=False,method='first'); out=[]
Ok=O.set_index(key+['cb'])
for k,g in Ln.groupby(key):
    if len(g)<5: continue
    g=g.sort_values('grk'); x=g.iloc[0]; rest=g.iloc[1:].sort_values('f3',ascending=False).head(3); hh=int(x.h)
    for a,b in itertools.permutations(sorted(rest.lane),2):
        cb=f'{hh}-{a}-{b}'
        try: od=Ok.loc[k+(cb,),'odds']
        except KeyError: continue
        out.append(k+(cb,float(od),x.res,x.gap,x.mo-rest.mo.mean(),x.er,rest.er.mean(),x.ph,hh))
T=pd.DataFrame(out,columns=key+['cb','odds','res','gap','modiff','xer','rer','ph','h']).merge(p3,on=key+['res']); T['pay']=np.where(T.cb==T.res,T.payout,0); T['i']=T.hd+T.jcd+T.rno.astype(str)
rep('売れすぎXを外した6点 全部',T,cut); rep('15倍未満',T[T.odds<15],cut); rep('30倍未満',T[T.odds<30],cut)
s=T[(T.gap>=.1)&(T.modiff<=-5)]; rep('売れすぎ幅0.1以上＆Xのモーター5以上低い',s,cut); rep('　＋30倍未満',s[s.odds<30],cut); rep('　＋15倍未満',s[s.odds<15],cut)
s2=s[(s.xer>=5)&(s.rer<s.xer)]; rep('　＋X展示5〜6位',s2,cut); rep('　　＋30倍未満',s2[s2.odds<30],cut); rep('　　＋15倍未満',s2[s2.odds<15],cut)
s3=T[(T.gap>=.1)&(T.xer>=5)&(T.rer<T.xer)]; rep('売れすぎ幅0.1以上＆X展示5〜6位（モーター条件なし）',s3,cut); rep('　＋30倍未満',s3[s3.odds<30],cut); rep('　＋頭60%以上＆30倍未満',s3[(s3.ph>=.6)&(s3.odds<30)],cut)
