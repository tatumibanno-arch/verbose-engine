"""別の型を同じやり方で探す。F2=買われなさすぎの相手Uを入れる、F3=①そのものが買われなさすぎ。
発見（〜2026-03）／検証（2026-04〜）。締切時オッズ（締切前とは違う）。"""
import sys, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
from sklearn.linear_model import LogisticRegression
R,L,oc=ob.load(); L=L.dropna(subset=['mo','nw']).copy()
X=lambda d:np.column_stack([d.r3,d.r2,d.mo/100,d.nw/10]+[(d.lane==k).astype(int) for k in range(3,7)])
disc=L[(L.hd<'20260401')&L.win1]; m=LogisticRegression(C=10,max_iter=500).fit(X(disc),disc.in3); L['f3']=m.predict_proba(X(L))[:,1]; L['gap']=L.mk3-L.f3
key=['hd','jcd','rno']; L['mo_rest']=(L.groupby(key).mo.transform('sum')-L.mo)/(L.groupby(key).mo.transform('size')-1)
E1=pd.read_parquet(ob.D+'ob_ent.parquet'); e1=E1[E1.lane==1][key+['er','mo','nw','cn','cw']].rename(columns={'er':'er1','mo':'mo1','nw':'nw1','cn':'cn1','cw':'cw1'})
def rep(nm,T):
    Rr=lambda s: round(s.pay.sum()/(100*len(s)),3) if len(s) else np.nan
    print(f'{nm} | 点 {len(T)} レース {T.rid.nunique()} 回収 {Rr(T)} 発見 {Rr(T[T.hd<"20260401"])} 検証 {Rr(T[T.hd>="20260401"])}')
# ---- F2
Rk=R.set_index(key)
U=L[(L.p1>=.6)&(L.gap<=-.10)].merge(e1[key+['er1']],on=key,how='left')
rows=[]
for r in U.itertuples():
    o=Rk.loc[(r.hd,r.jcd,r.rno)]; u=r.lane
    for y in range(2,7):
        if y==u: continue
        for cb,ok in ((f'1-{u}-{y}',f'o{u}{y}'),(f'1-{y}-{u}',f'o{y}{u}')):
            rows.append((r.hd,r.jcd,r.rno,u,cb,o[ok],o.res,o.payout,r.gap,r.mo-r.mo_rest,r.er,r.p1,r.er1,cb[2]==str(u)))
T=pd.DataFrame(rows,columns=key+['u','cb','odds','res','payout','gap','modiff','uer','p1','er1','u2nd']); T['pay']=np.where(T.cb==T.res,T.payout,0); T['rid']=T.hd+T.jcd+T.rno.astype(str)+T.u.astype(str)
print('== F2: 買われなさすぎの相手U（売れ方が見立てより0.10以上低い）を2・3着に入れる、①頭60%以上'); rep('全部（8点）',T)
for a in (15,30,60): rep(f'{a}倍未満',T[T.odds<a])
g=T[(T.modiff>=5)]; rep('Uのモーターが他より5以上高い',g); rep('　＋U展示1〜2位',g[g.uer<=2]); rep('　＋U展示1〜2位＆30倍未満',g[(g.uer<=2)&(g.odds<30)]); rep('　＋U展示1〜2位＆15倍未満',g[(g.uer<=2)&(g.odds<15)])
rep('U展示1位',T[T.uer==1]); rep('U展示1位＆30倍未満',T[(T.uer==1)&(T.odds<30)]); rep('U展示1位＆Uが2着の組',T[(T.uer==1)&T.u2nd]); rep('U展示1位＆Uが3着の組',T[(T.uer==1)&~T.u2nd])
rep('売れなさすぎ幅0.2以上＆U展示1〜2位',T[(T.gap<=-.2)&(T.uer<=2)]); rep('　＋30倍未満',T[(T.gap<=-.2)&(T.uer<=2)&(T.odds<30)])
rep('①頭70%以上＆Uモーター5以上高い＆U展示1〜2位＆30倍未満',g[(g.p1>=.7)&(g.uer<=2)&(g.odds<30)])
# ---- F3
A=R[key+['res','payout','tot']+oc].merge(e1,on=key).dropna(subset=['mo1','nw1','er1']); inv=1/A[oc]; A['p1']=inv.sum(1)/A.tot; A['win1']=A.res.str[0]=='1'
b=A[A.win1].shape[0]/len(A); A['c1w']=(A.cw1.fillna(0)+10*b)/(A.cn1.fillna(0)+10)
X1=lambda d:np.column_stack([d.c1w,pd.to_numeric(d.mo1)/100,pd.to_numeric(d.nw1)/10,d.er1])
tr=A[A.hd<'20260401']; m1=LogisticRegression(C=10,max_iter=500).fit(X1(tr),tr.win1); A['f1']=m1.predict_proba(X1(A))[:,1]; A['g1']=A.f1-A.p1
rows=[]
for r in A[A.g1>=.05].itertuples():
    cs=sorted([(getattr(r,c),c) for c in oc])
    for k,(o,c) in enumerate(cs,1):
        if k<=6: rows.append((r.hd,r.jcd,r.rno,k,o,f'1-{c[1]}-{c[2]}',r.res,r.payout,r.g1,r.p1,r.er1,r.f1))
T3=pd.DataFrame(rows,columns=key+['rk','odds','cb','res','payout','g1','p1','er1','f1']); T3['pay']=np.where(T3.cb==T3.res,T3.payout,0); T3['rid']=T3.hd+T3.jcd+T3.rno.astype(str)
print('\n== F3: ①が成績・展示の見立てより売れていない（差0.05以上）。①頭の安い順に買う'); rep('安い順6点',T3)
for k in (1,2,3): rep(f'安い順{k}点',T3[T3.rk<=k])
for a,bb in ((.05,.1),(.1,.15),(.15,.2),(.2,1)): rep(f'差 {a}〜{bb} 安い順3点',T3[(T3.g1>=a)&(T3.g1<bb)&(T3.rk<=3)])
rep('差0.1以上＆①展示1位 安い順3点',T3[(T3.g1>=.1)&(T3.er1==1)&(T3.rk<=3)]); rep('差0.1以上＆①展示1〜2位＆①頭50%以上 安い順3点',T3[(T3.g1>=.1)&(T3.er1<=2)&(T3.p1>=.5)&(T3.rk<=3)])
rep('差0.1以上＆15倍未満',T3[(T3.g1>=.1)&(T3.odds<15)]); rep('差0.1以上＆①頭40〜60% 安い順3点',T3[(T3.g1>=.1)&(T3.p1>=.4)&(T3.p1<.6)&(T3.rk<=3)])
