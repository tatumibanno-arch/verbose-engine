"""3つの知らせの同じレースで、買い方（券種・点数）を変えたら回収率はどうか。確定払戻（締切前オッズとは違う）。"""
import sqlite3, numpy as np, pandas as pd, itertools, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/three_signals.py').read().split("R=lambda s,col")[0])
cut='20260201'
P=pd.read_sql("select hd,jcd,rno,ticket_type tt,combination cb_,payout from payouts where hd>='20250601'",c)
P['k']=P.hd+P.jcd+P.rno.astype(str)
def norm(tt,x):
    x=x.replace('=','-')
    return '-'.join(sorted(x.split('-'))) if tt in('nirenfuku','wide','sanrenfuku') else x
P['n']=[norm(t,x) for t,x in zip(P.tt,P.cb_)]
pay={}
for r in P.itertuples(): pay[(r.k,r.tt,r.n)]=pay.get((r.k,r.tt,r.n),0)+r.payout if r.tt!='wide' else r.payout
# 3rd boat c: best lap among others
Lp=X[key+['lane','lap']].dropna(subset=['lap']).copy(); Lp['k']=Lp.hd+Lp.jcd+Lp.rno.astype(str)
lapd={k:g.sort_values('lap').lane.astype(int).tolist() for k,g in Lp.groupby('k')}
S=S.copy(); S['k']=S.hd+S.jcd+S.rno.astype(str)
S['a']=S.cb.str.split('-').str[1]; S['b']=S.cb.str.split('-').str[2]
def third(r):
    for l in lapd.get(r.k,[]):
        if str(l) not in('1',r.a,r.b): return str(l)
    return None
S['c']=[third(r) for r in S.itertuples()]
def V(r):
    k,a,b,c=r.k,r.a,r.b,r.c
    g=lambda t,n: pay.get((k,t,n),0)
    s=lambda *x:'-'.join(sorted(x))
    o={}
    o['tansho1']=(1,g('tansho','1'))
    o['2tan 1-a']=(1,g('nirensho',f'1-{a}'))
    o['2tan 1-a,b']=(2,g('nirensho',f'1-{a}')+g('nirensho',f'1-{b}'))
    o['2fuku 1=a']=(1,g('nirenfuku',s('1',a)))
    o['wide 1-a']=(1,g('wide',s('1',a)))
    o['3fuku 1-a-b']=(1,g('sanrenfuku',s('1',a,b)))
    o['3tan 1-ab-ab']=(2,g('sanrensho',f'1-{a}-{b}')+g('sanrensho',f'1-{b}-{a}'))
    o['3tan 1-ab-all']=(8,sum(g('sanrensho',f'1-{x}-{y}') for x in(a,b) for y in'23456' if y!=x))
    if c: o['3tan 1-abc-abc']=(6,sum(g('sanrensho',f'1-{x}-{y}') for x in(a,b,c) for y in(a,b,c) if x!=y))
    o['3tan 1pt(now)']=(1,g('sanrensho',r.cb))
    return o
def evalset(D,name):
    rows=[V(r) for r in D.itertuples()]
    out=[]
    for v in rows[0].keys():
        idx=[i for i,r in enumerate(rows) if v in r]; pts=rows[idx[0]][v][0]
        d=D.iloc[idx].copy(); d['pt']=[rows[i][v][0] for i in idx]; d['py']=[rows[i][v][1] for i in idx]
        d=d.sort_values(key)
        roi=lambda x: x.py.sum()/(100*x.pt.sum()) if len(x) else np.nan
        mo=d.groupby(d.hd.str[:6]).apply(roi)
        h=d.py.sort_values(ascending=False).head(3).sum(); ex=(d.py.sum()-h)/(100*d.pt.sum())
        cum=np.cumsum(d.py.values-100*d.pt.values); dd=(cum-np.maximum.accumulate(np.maximum(cum,0))).min()
        out.append((v,pts,len(d),roi(d),roi(d[d.hd<cut]),roi(d[d.hd>=cut]),ex,f'{(mo>1).sum()}/{len(mo)}',int(dd)))
    print(f'## {name} races={len(D)}'); print(f'{"variant":16}pts n    ROI   前半  後半  exTop3 win-mo  maxDD')
    for o in out: print(f'{o[0]:16}{o[1]:<4}{o[2]:<5}{o[3]:.3f} {o[4]:.3f} {o[5]:.3f} {o[6]:.3f} {o[7]:6} {o[8]}')
for sg,s in S.groupby('sig'): evalset(s,sg)
evalset(S.drop_duplicates(key+['cb']).drop_duplicates('k'),'union')
