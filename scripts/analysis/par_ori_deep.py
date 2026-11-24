"""par_ori_deep: oriten 深掘り。事前に決めた19仮説（閾値は分位で固定、ROIは見てから変えない）。オッズ不使用。
H1 ①一周1位かつ 1位-2位差が上位25%  H2 ①1位かつ(lap4-lap3)/場の標準偏差が S1と同数になる閾値(2,3位は問わず)
H3 ①一周1位かつ まわり足でも①と2位の差が上位25%  H4 同 直線
H5 ①一周1位 & 一周2-3位に勝率順位>=4の艇  H6 同 公式展示順位>=4の艇
H7 ①一周1位 & ①の勝率順位>=3  H8 ①一周4位以下で、非①の一周1位x: x-1-y(y=一周次点)  H9 同 x-y-1
H10 ①が一周・まわり足・直線の全部1位  H11 非①が3つ全部1位: x-1-y
H12-16 S1内の2,3着の順を 一周/まわり足/直線/スタ展/勝率 で決める  H17 常に1-2-3  H18 S1で両順2点
基準: 同じ作り方の買い目を全レースで。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
X=X.sort_values(key+['lane']).reset_index(drop=True)
g=X.groupby(key,sort=False); R0=g.size().reset_index()[key]; N=len(R0)
def M(col): 
    a=np.full((N,6),np.nan); idx=np.repeat(np.arange(N),g.size().values)  # lanes complete? check
    return a
# 6艇揃いだけ
cnt=g.size(); full=cnt[cnt==6].index; X=X.set_index(key).loc[full].reset_index(); 
N=X.groupby(key,sort=False).ngroups; R0=X.groupby(key,sort=False).size().reset_index()[key]
A=lambda c: X[c].values.reshape(N,6)
lap,maw,cho,et,st,nw,rl,rm,rc,rx,rs,rn=[A(c) for c in ['lap','mawari','chokusen','et','st','nw','r_lap','r_maw','r_cho','r_ex','r_st','r_nw']]
hd=R0.hd.values; ar=np.arange(N)
ok=~np.isnan(lap).any(1)&~np.isnan(et).any(1)
ordL=np.argsort(np.where(np.isnan(lap),99,lap),1)  # lane idx sorted by lap
def srt(a): return np.take_along_axis(a,ordL,1)
lapS=srt(lap)
pm=p.set_index(key); res=pm.reindex(pd.MultiIndex.from_frame(R0)).res.values; pay=pm.reindex(pd.MultiIndex.from_frame(R0)).payout.values
has=~pd.isna(res)
def tk(*ix): return np.array([f'{a+1}-{b+1}-{c+1}' for a,b,c in zip(*ix)])
def pair_ticket(a,b,keycol):  # 1-a-b, 順は keycol(小さい方先)
    ka=keycol[ar,a]; kb=keycol[ar,b]; f=(ka<kb)|((ka==kb)&(a<b)) if True else None; return np.where(f,tk(np.zeros(N,int),a,b),tk(np.zeros(N,int),b,a))
def two_tickets(a,b): return [tk(np.zeros(N,int),a,b),tk(np.zeros(N,int),b,a)]
L=ordL; h1,h2,h3,h4=L[:,0],L[:,1],L[:,2],L[:,3]
gap34=lapS[:,3]-lapS[:,2]; m12=lapS[:,1]-lapS[:,0]
cut_=hd>='20260201'
def stat(nm,m,tick,base=None,show=True):
    m=m&ok&has; tick=tick if isinstance(tick,list) else [tick]
    pp=np.zeros(N)
    for t in tick: pp+=np.where(res==t,pay,0)
    cost=100*len(tick)
    s=m; n=s.sum()
    if n<30: print(f'{nm:42s} n={n} 少ない'); return None
    mo=pd.Series(pp[s]).groupby(pd.Series(hd[s]).str[:6].values).sum()/ (cost*pd.Series(hd[s]).str[:6].value_counts().sort_index())
    rr=lambda q: pp[q].sum()/(cost*q.sum()) if q.sum() else np.nan
    srt_=np.sort(pp[s])[::-1]; ex=lambda k: (pp[s].sum()-srt_[:k].sum())/(cost*n)
    hits=(pp[s]>0).sum()
    out=dict(nm=nm,n=n,hit=hits,roi=rr(s),h1=rr(s&~cut_),h2=rr(s&cut_),x1=ex(1),x3=ex(3),x5=ex(5),wm=f'{(mo>1).sum()}/{len(mo)}')
    if show: print(f"{nm:42s} n={n:5d} hit={hits:4d}({hits/n:.3f}) ROI={out['roi']:.3f} 前{out['h1']:.2f} 後{out['h2']:.2f} 除1/3/5={out['x1']:.2f}/{out['x3']:.2f}/{out['x5']:.2f} 勝月{out['wm']}")
    return out,pp,s
def fake(nm,pp,s,basem,cost=100,B=3000):
    rng=np.random.default_rng(1); pool=np.where(basem&ok&has)[0]; n=s.sum(); obs=pp[s].sum()/(cost*n)
    sims=np.array([pp[rng.choice(pool,n,replace=False)].sum()/(cost*n) for _ in range(B)])
    print(f'    fake(同じ作り方の全レースから同数ランダム) 観測{obs:.3f} 上位{(sims>=obs).mean():.4f} 疑似平均{sims.mean():.3f} 95%点{np.quantile(sims,.95):.3f}')
tried=0; C=[]
def T(nm,m,tick,basem):
    global tried; tried+=1
    r=stat(nm,m,tick)
    bl=stat('   基準(同じ作り方 全レース)',basem,tick)
    if r and bl is None: pass
    if r: C.append((nm,r,basem))
    return r
# ---- S1 再現
l1=(h1==0)
S1m=l1&np.isin(h2,[1,2])&np.isin(h3,[1,2])&(gap34>=0.4)
tS1=pair_ticket(h2,h3,et)
print('S1再現'); stat('S1 (lap-one)',S1m,tS1); stat('S1の基準 ①一周1位 1-a-b',l1,tS1)
# ---- A
q=lambda x,m,p_=75: np.nanpercentile(x[m&ok],p_)
T('H1 ①1位&1-2位差上位25%',l1&(m12>=q(m12,l1)),tS1,l1)
# H2 場ごとのsd
Xd=X.copy(); Xd['dm']=Xd.lap-Xd.groupby(key).lap.transform('mean'); vs=Xd.groupby('jcd').dm.std(); vsd=R0.jcd.map(vs).values
z34=gap34/vsd; thr=np.nanpercentile(z34[l1&ok],100*(1-S1m.sum()/ (l1&ok).sum()))
print('  H2閾値(z)',round(thr,3),'S1同数'); T('H2 ①1位&gap34/場sd(S1と同数)',l1&(z34>=thr),tS1,l1)
mS=np.sort(np.where(np.isnan(maw),99,maw),1); cS=np.sort(np.where(np.isnan(cho),99,cho),1)
mm=np.where(rm[:,0]==1,1,0)  # ①のまわり足1位
mmarg=mS[:,1]-maw[:,0]; mgm=l1&(rm[:,0]==1)
T('H3 ①一周&まわり1位&まわり差上位25%',mgm&(mmarg>=q(mmarg,mgm)),tS1,l1&(rm[:,0]==1))
cmarg=cS[:,1]-cho[:,0]; cg=l1&(rc[:,0]==1)
T('H4 ①一周&直線1位&直線差上位25%',cg&(cmarg>=q(cmarg,cg)),tS1,cg)
# ---- B
topn=lambda r,k: (r<=k)
und=lambda rr_: (topn(rl,3)&(rr_>=4)&(np.arange(6)[None,:]>0))  # 非①で一周top3かつ rr>=4
for nm,rr_ in (('H5 勝率順位>=4',rn),('H6 公式展示順位>=4',rx)):
    u=und(rr_); cond=l1&(u.sum(1)>=1)
    # 一周top3の非①: 候補2艇
    a,b=h2,h3
    T(f'{nm}の艇が一周2-3位に(①1位)',cond,tS1,l1)
T('H7 ①一周1位&①勝率順位>=3',l1&(rn[:,0]>=3),tS1,l1)
# H8/H9
xh=h1; cond=(rl[:,0]>=4)&(xh!=0)
y=np.where(h2==0,h3,h2)  # 一周次点(①以外)
z=np.zeros(N,int)
T('H8 ①一周4位以下 x-1-y',cond,tk(xh,z,y),(xh!=0))
T('H9 ①一周4位以下 x-y-1',cond,tk(xh,y,z),(xh!=0))
# ---- C
agree=(rl[:,0]==1)&(rm[:,0]==1)&(rc[:,0]==1)
T('H10 ①が一周・まわり・直線全部1位',agree,tS1,l1)
aa=(rl==1)&(rm==1)&(rc==1); aa[:,0]=False; xa=aa.argmax(1); cond=aa.any(1)
ya=np.where(h2==xa,h3,h2); ya=np.where(ya==xa,h4,ya)
T('H11 非①が3つ全部1位 x-1-y',cond,tk(xa,z,ya),(xh!=0))
# ---- D (S1内)
for nm,col in (('H12 2,3着=一周順',lap),('H13 まわり足順',maw),('H14 直線順',cho),('H15 スタ展順',st),('H16 勝率高い順',-nw)):
    T(nm,S1m,pair_ticket(h2,h3,np.where(np.isnan(col),9,col)),S1m)
T('H17 常に1-2-3',S1m,np.array(['1-2-3']*N),S1m)
T('H18 両順2点',S1m,two_tickets(h2,h3),S1m)
print('試した仮説数',tried)
print('\n== ROI>1.1 が前後半とも なもの fake check')
for nm,r,basem in C:
    o,pp,s=r
    if o['h1']>1.1 and o['h2']>1.1:
        print(nm); 
        fake(nm,pp,s,basem,100 if True else 200)
print('\n== 事後の確認（探索扱い・採用根拠にしない）S1の骨格を分解')
l23=l1&np.isin(h2,[1,2])&np.isin(h3,[1,2])
stat('X1 ①1位,②③が2,3位, 差なし',l23,tS1)
for g_ in (0.2,0.3,0.4,0.5,0.6): stat(f'X2 同 gap34>={g_}',l23&(gap34>=g_),tS1)
stat('X3 ①1位 gap34>=0.4 (2,3位の艇は問わず)',l1&(gap34>=0.4),tS1)
stat('X4 ①1位 gap34>=0.4 かつ ②③以外を含む',l1&(gap34>=0.4)&~l23,tS1)
