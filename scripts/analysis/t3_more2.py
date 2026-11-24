"""3連単の別の形（オッズを使わない）。点数＝7つの順位の合計。
A ①が抜けている（点数1位と2位の差が大きい）とき、1点（2・3着=点数2・3位、順は公式展示）／いつも1-2-3／2着固定・3着流し
B 2着の艇が抜けている（点数2位と3位の差が大きい）とき、①-2位-流し（4点）
C 知らせ1・2の手前（差が少し足りない）レースで、条件を2つとも少しずつ満たす"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2']; X['sc']=X[cols].sum(1)
Y=X.sort_values(key+['sc','lane']); Y['o']=Y.groupby(key).cumcount()+1
T=Y[Y.o<=6].pivot_table(index=key,columns='o',values=['lane','sc','et'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index(); T=T[T.lane1==1].merge(p,on=key)
T['g12']=T.sc2-T.sc1; T['g23']=T.sc3-T.sc2; T['g34']=T.sc4-T.sc3; L=lambda a: a.astype(int).astype(str)
a,b_=L(T.lane2),L(T.lane3); f=T.et2<=T.et3; T['cb1']='1-'+np.where(f,a,b_)+'-'+np.where(f,b_,a)
def rep(nm,s,hitfn,npts):
    pay=pd.Series([hitfn(r) for r in s.itertuples()],index=s.index); m=lambda q: pay[q.index].sum()/(100*npts*len(q)) if len(q) else np.nan; mo=s.groupby(s.hd.str[:6]).apply(m)
    print(f'{nm} | レース {len(s)} 的中 {(pay>0).mean():.3f} 回収 {m(s):.3f} 前半 {m(s[s.hd<cut]):.3f} 後半 {m(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
print('== A ①が抜けている（1位と2位の差）')
for g in (6,9,12):
    s=T[T.g12>=g]; rep(f'差{g}以上 1点（点数2・3位、公式展示順）',s,lambda r: r.payout if r.res==r.cb1 else 0,1)
    rep(f'  いつも1-2-3',s,lambda r: r.payout if r.res=='1-2-3' else 0,1)
    rep(f'  ①-点数2位-流し（4点）',s,lambda r: r.payout if r.res[:3]==f'1-{int(r.lane2)}' else 0,4)
print('== B 2着の艇が抜けている（2位と3位の差）、①-2位-流し（4点）')
for g in (4,6,8): s=T[T.g23>=g]; rep(f'差{g}以上',s,lambda r: r.payout if r.res[:3]==f'1-{int(r.lane2)}' else 0,4)
print('== C 3位と4位の差が少し足りない（7〜9）レース')
s=T[T.g34.between(7,9.99)]; rep('差7〜9 1点',s,lambda r: r.payout if r.res==r.cb1 else 0,1)
s2=s[(s.g12>=3)]; rep('  ＋①も抜けている（1位と2位の差3以上）',s2,lambda r: r.payout if r.res==r.cb1 else 0,1)
