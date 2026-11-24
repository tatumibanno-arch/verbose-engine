"""par_trend_: 節内の変化（同じ場・同じモーターの前走からの展示の上下、前走の着順との食い違い）。オッズ不使用。
仮説14本を先に決めてから回す。払戻は確定。前半〜2026-01／後半2026-02〜。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
# ---- 節の組み立て
e=pd.read_sql("select hd,jcd,rno,lane,racer_no,motor_no from entries where hd>='20250515'",c)
r=pd.read_sql("select hd,jcd,rno,lane,position from results where hd>='20250515'",c)
r['position']=pd.to_numeric(r.position,errors='coerce')
e=e.merge(r,on=key+['lane'],how='left'); e['position']=e.position.fillna(6)
z=X[key+['lane','lap','mawari','chokusen']].copy()
for q in ('lap','mawari','chokusen'):
    g=z.groupby(key)[q]; z['z_'+q]=(z[q]-g.transform('mean'))/g.transform('std').replace(0,np.nan)
z['zc']=z[['z_lap','z_mawari','z_chokusen']].mean(1)
e=e.merge(z[key+['lane','zc','z_lap']],on=key+['lane'],how='left')
e['d']=pd.to_datetime(e.hd).map(pd.Timestamp.toordinal)
e=e.sort_values(['jcd','racer_no','d','rno']).reset_index(drop=True)
gk=['jcd','racer_no']
new=(e.groupby(gk).d.diff()>2)|(e.groupby(gk).motor_no.shift()!=e.motor_no)|(e.groupby(gk).d.diff().isna())
e['meet']=new.cumsum()
g=e.groupby('meet')
e['has']=e.zc.notna().astype(int); z0=e.zc.fillna(0)
e['n_prev']=g.has.cumsum()-e.has; e['s_prev']=z0.groupby(e.meet).cumsum()-z0
e['pz']=e.s_prev/e.n_prev.replace(0,np.nan)
e['np_pos']=g.cumcount(); e['pos_prev']=(e.position.groupby(e.meet).cumsum()-e.position)/e.np_pos.replace(0,np.nan)
e['last_zc']=g.zc.shift(); e['last_d']=g.d.shift(); e['last_pos']=g.position.shift()
e['dz']=e.zc-e.pz
e['dz_same']=np.where(e.last_d==e.d,e.zc-e.last_zc,np.nan)
e=e[e.hd>='20250601']
T=X[key+['lane','et','r_lap','r_nw']].merge(e[key+['lane','n_prev','pz','pos_prev','dz','dz_same','last_pos','zc']],on=key+['lane'],how='left')
T['rz']=T.groupby(key).zc.rank(method='average')   # 今日の総合順位(1=良)
print('boat-rows',len(T),'with n_prev>=2',(T.n_prev>=2).sum())
# ---- 艇ごと→レース横持ち
W={}
for q in ('zc','pz','n_prev','pos_prev','dz','dz_same','rz','et','r_lap'):
    W[q]=T.pivot_table(index=key,columns='lane',values=q,aggfunc='first',dropna=False)
idx=W['et'].index.union(W['zc'].index); N=len(idx)
A={q:W[q].reindex(index=idx,columns=range(1,7)).values for q in W}
zc,pz,npv,pp,dz,dzs,rz,et=A['zc'],A['pz'],A['n_prev'],A['pos_prev'],A['dz'],A['dz_same'],A['rz'],A['et']
ok=np.isfinite(zc).sum(1)>=5
def best_non1(excl=()):
    """今日の総合が良い順(①除く、excl艇も除く)。返値: 艇番の配列(N,5)順位順"""
    m=zc.copy(); m[:,0]=np.inf
    for ex in excl: m[np.arange(N),ex]=np.inf
    m=np.where(np.isfinite(m),m,np.inf)
    return np.argsort(m,1,kind='stable')
def tickets(head,second,third):  # 0始まり艇番→'a-b-c'
    return pd.Series([f'{a+1}-{b+1}-{c+1}' for a,b,c in zip(head,second,third)],index=idx)
def tail12():
    o=best_non1()[:,:2]; a,b=o[:,0],o[:,1]; ar=np.arange(N); f=et[ar,a]<=et[ar,b]
    return np.where(f,a,b),np.where(f,b,a)
t2,t3=tail12(); ar=np.arange(N)
one=np.zeros(N,int)
T1=tickets(one,t2,t3)                      # 1-x-y
o3=best_non1(); A1,B1=o3[:,0],o3[:,1]
H={}
def add(name,mask,tk,base): H[name]=(np.asarray(mask)&ok,tk,np.asarray(base)&ok)
n1=npv[:,0]>=2; p1=pz[:,0]; d1=dz[:,0]; q1=pp[:,0]
# 1-4: ①の変化
add('H1 ①好転(dz<=-0.7)かつ今日1位 →1-x-y',n1&(d1<=-0.7)&(rz[:,0]<=1.5),T1,n1&(rz[:,0]<=1.5))
add('H2 ①好転(dz<=-0.7) →1-x-y',n1&(d1<=-0.7),T1,n1)
add('H3 ①悪化(dz>=0.7)かつ前は良(pz<=-0.3) →A-B-1',n1&(d1>=0.7)&(p1<=-0.3),tickets(A1,B1,one),n1)
add('H4 ①悪化(dz>=0.7)かつ前着順良(平均<=2.5) →A-1-B',n1&(d1>=0.7)&(q1<=2.5),tickets(A1,one,B1),n1)
# 5: ①は展示良(pz<=-0.5)なのに前走の着が悪い(平均>=3.5)=運が悪かった
add('H5 ①展示良なのに前着順悪 →1-x-y',n1&(p1<=-0.5)&(q1>=3.5),T1,n1)
# 6: ①展示悪(pz>=0.5)なのに着順は良(<=2.5)=運が良かっただけ → ①外し
add('H6 ①展示悪なのに前着順良 →A-B-1',n1&(p1>=0.5)&(q1<=2.5),tickets(A1,B1,one),n1)
# 7-8: 外の艇の好転
nb=(npv>=2); nb[:,0]=False
dzm=np.where(nb&np.isfinite(dz),dz,np.inf); j=dzm.argmin(1); dmin=dzm.min(1)
add('H7 最も好転した外艇(dz<=-1.0)が今日1位 →X-1-y',(dmin<=-1.0)&(rz[ar,j]<=1.5),
    tickets(j,one,np.where(B1==j,A1,B1) if False else np.where(A1==j,B1,A1)),(dmin<np.inf))
add('H8 同上(dz<=-1.0, 順位問わず) →1-X-y',(dmin<=-1.0),
    tickets(one,j,np.where(A1==j,B1,A1)),(dmin<np.inf))
# 9-10: 外の艇は節を通して展示1位級(pz<=-0.7)なのに着が悪い(平均>=4)＝皆は着順で見送り
pq=np.where(nb&(pz<=-0.7)&(pp>=4),pz,np.inf); k=pq.argmin(1); hasq=pq.min(1)<np.inf
add('H9 外の前展示上位×前着順悪の艇X →1-X-y',hasq&(npv[:,0]>=0),tickets(one,k,np.where(A1==k,B1,A1)),np.ones(N,bool)&hasq)
add('H10 同上 Xを3着に →1-y-X',hasq,tickets(one,np.where(A1==k,B1,A1),k),hasq)
# 11: 同じ日の前走から大きく好転(同日・再調整)した①
dsm=np.isfinite(dzs[:,0])
add('H11 ①が同日前走より好転(dz_same<=-0.8) →1-x-y',dsm&(dzs[:,0]<=-0.8),T1,dsm)
# 12: 外の艇が同日前走より好転 → 頭
dsb=np.isfinite(dzs); dsb[:,0]=False
dsm2=np.where(dsb,dzs,np.inf); j2=dsm2.argmin(1)
add('H12 外艇が同日前走より好転(<=-1.0) →1-X-y',dsm2.min(1)<=-1.0,tickets(one,j2,np.where(A1==j2,B1,A1)),dsm2.min(1)<np.inf)
# 13: 節の平均展示で①が最良かつ今日も良(安定)
pm=np.where(np.isfinite(pz)&(npv>=2),pz,np.inf); pm1=pm[:,0]; others=pm[:,1:].min(1)
add('H13 ①が節平均展示で最良(他より0.5以上) →1-x-y',(pm1<np.inf)&(others-pm1>=0.5)&(others<np.inf),T1,(pm1<np.inf)&(others<np.inf))
# 14: ①の前走着順が悪い(最終走>=5)が節平均展示は上位
add('H14 ①前走5着以下だが節展示は上位(pz<=-0.3) →1-x-y',n1&(pp[:,0]>-1)&(A['pos_prev'][:,0]>0)&(p1<=-0.3)&(T[T.lane==1].set_index(key).reindex(idx).last_pos.values>=5),T1,n1)
print('試した数',len(H))
P=p.set_index(key).reindex(idx); res=P.res.values; pay=P.payout.values
hd=idx.get_level_values('hd').values; mon=np.array([s[:6] for s in hd]); fh=hd<cut
def stat(m,tk):
    v=np.where(m&np.isfinite(pay)&(res!=None),1,0).astype(bool)
    hit=(tk.values==res)&v; y=np.where(hit,pay,0.0)
    return v,y
def line(nm,m,tk,base,S_keys):
    v,y=stat(m,tk); n=v.sum()
    if n<30: print(nm,'n',n,'少ない'); return
    ys=y[v]; h=ys>0; s=np.sort(ys)[::-1]; tot=ys.sum()
    ex=lambda k:(tot-s[:k].sum())/(100*n)
    f=v&fh; l=v&~fh
    mo=pd.Series(y[v]).groupby(mon[v]).apply(lambda s:s.sum()/(100*len(s)))
    vb,yb=stat(base,tk); bn=vb.sum()
    ov=np.array([k in S_keys for k in idx[v]]).mean()
    print(f'{nm} | n{n} 的中{h.mean():.3f} 回収{tot/100/n:.3f} 前{y[f].sum()/100/max(f.sum(),1):.3f}(n{f.sum()}) 後{y[l].sum()/100/max(l.sum(),1):.3f}(n{l.sum()}) 除1/3/5 {ex(1):.2f}/{ex(3):.2f}/{ex(5):.2f} 月{(mo>1).sum()}/{len(mo)} 同券基準{yb[vb].sum()/100/bn:.3f}(n{bn}) 知らせ重なり{ov:.2f}')
# three_signals の重なり
ts=open('scripts/analysis/three_signals.py').read().split("R=lambda s,col='pay'")[0].split("\n",3)[3]
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()): exec(ts)
S_keys=set(map(tuple,S[key].values))
for nm,(m,tk,b) in H.items(): line(nm,m,tk,b,S_keys)
