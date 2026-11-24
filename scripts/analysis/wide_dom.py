"""拡連複（ワイド）と複勝で「上位が抜けている」度合いを使う。点数＝7つの順位の合計。①が点数1位。オッズは使わない。払戻は確定。
A ワイド1点 ①-点数2位 ／ B ワイド1点 点数2位-3位 ／ C ワイド3点（上位3艇の組み合わせ全部） ／ D 複勝①"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2']; X['sc']=X[cols].sum(1)
Y=X.sort_values(key+['sc','lane']); Y['o']=Y.groupby(key).cumcount()+1
T=Y[Y.o<=4].pivot_table(index=key,columns='o',values=['lane','sc'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index(); T=T[T.lane1==1]
T['g23']=T.sc3-T.sc2; T['g34']=T.sc4-T.sc3
w=pd.read_sql("select hd,jcd,rno,combination cb,payout from payouts where ticket_type='wide' and hd>='20250601'",c); w['cb']=w.cb.str.replace('=','-').apply(lambda x:'-'.join(sorted(x.split('-'))))
wd={}
for r in w.itertuples(): wd.setdefault((r.hd,r.jcd,r.rno),{})[r.cb]=r.payout
f=pd.read_sql("select hd,jcd,rno,combination cb,payout from payouts where ticket_type='fukusho' and hd>='20250601'",c); fd={}
for r in f.itertuples(): fd.setdefault((r.hd,r.jcd,r.rno),{})[str(r.cb)]=r.payout
L=lambda x: str(int(x))
def wide(s,pairs_fn):
    pay=[];n=[]
    for r in s.itertuples():
        d=wd.get((r.hd,r.jcd,r.rno),{}); ps=pairs_fn(r); pay.append(sum(d.get('-'.join(sorted(p)),0) for p in ps)); n.append(len(ps))
    return np.array(pay),np.array(n)
def rep(nm,s,pay,n):
    s=s.assign(pay=pay,n=n); m=lambda q: q.pay.sum()/(100*q.n.sum()) if len(q) else np.nan; mo=s.groupby(s.hd.str[:6]).apply(m)
    print(f'{nm} | レース {len(s)} 的中 {(s.pay>0).mean():.3f} 回収 {m(s):.3f} 前半 {m(s[s.hd<cut]):.3f} 後半 {m(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
for g in (0,6,8,10):
    s=T[T.g34>=g]
    rep(f'差（3位と4位）{g}以上 A ワイド ①-2位',s,*wide(s,lambda r:[('1',L(r.lane2))]))
    rep(f'  B ワイド 2位-3位',s,*wide(s,lambda r:[(L(r.lane2),L(r.lane3))]))
    rep(f'  C ワイド3点',s,*wide(s,lambda r:[('1',L(r.lane2)),('1',L(r.lane3)),(L(r.lane2),L(r.lane3))]))
for g in (0,4,6,8):
    s=T[T.g23>=g]; rep(f'差（2位と3位）{g}以上 A ワイド ①-2位',s,*wide(s,lambda r:[('1',L(r.lane2))]))
for g in (0,6,9,12):
    s=T[(T.sc2-T.sc1)>=g]; pay=np.array([fd.get((r.hd,r.jcd,r.rno),{}).get('1',0) for r in s.itertuples()]); rep(f'D 複勝① 1位と2位の差{g}以上',s,pay,np.ones(len(s)))
