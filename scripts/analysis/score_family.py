"""7つの順位の合計（t3-score-one-v1 の点数）の仲間。オッズは使わない。2025-06〜2026-11、前半〜2026-01／後半2026-02〜。
A 頭が①以外でも点数1位なら同じ買い方 ／ B 3連複1点（上位3艇） ／ C 2連単1点（点数1位-2位、2位と3番手の差） ／ D 単勝（点数1位、2番手との差）"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2']; Y=X.copy(); Y['sc']=Y[cols].sum(1); Y=Y.sort_values(key+['sc','lane']); Y['ord']=Y.groupby(key).cumcount()+1
T=Y[Y.ord<=4].pivot_table(index=key,columns='ord',values=['lane','sc','et'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index()
for t in ('sanrensho','sanrenfuku','nirensho','tansho'):
    q=pd.read_sql(f"select hd,jcd,rno,combination r_{t},payout p_{t} from payouts where ticket_type='{t}' and hd>='20250601'",c).drop_duplicates(key); T=T.merge(q,on=key,how='left')
T['g12']=T.sc2-T.sc1; T['g23']=T.sc3-T.sc2; T['g34']=T.sc4-T.sc3
L=lambda a: a.astype(int).astype(str)
def rep(nm,s,t,cb,sort=False):
    s=s.dropna(subset=[f'r_{t}']).copy(); cb=cb.loc[s.index]; res=s[f'r_{t}'].str.replace('=','-')
    if sort: res=res.apply(lambda x:'-'.join(sorted(x.split('-')))); cb=cb.apply(lambda x:'-'.join(sorted(x.split('-'))))
    s['pay']=np.where(res==cb,s[f'p_{t}'],0); mo=s.groupby(s.hd.str[:6]).apply(R)
    print(f'{nm} | レース {len(s)} 的中 {(s.pay>0).mean():.3f} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
t3=lambda s: L(s.lane1)+'-'+np.where(s.et2<=s.et3,L(s.lane2),L(s.lane3))+'-'+np.where(s.et2<=s.et3,L(s.lane3),L(s.lane2))
print('== A 3連単1点、頭=点数1位（艇番は問わない）')
for g in (9,10,11):
    s=T[T.g34>=g]; rep(f'差{g}以上 全部',s,'sanrensho',t3(s)); rep('  頭が①以外',s[s.lane1!=1],'sanrensho',t3(s[s.lane1!=1]))
print('== B 3連複1点（点数上位3艇）')
for g in (8,9,10,11):
    s=T[T.g34>=g]; rep(f'差{g}以上',s,'sanrenfuku',L(s.lane1)+'-'+L(s.lane2)+'-'+L(s.lane3),True); u=s[s.lane1==1]; rep('  ①が1位',u,'sanrenfuku',L(u.lane1)+'-'+L(u.lane2)+'-'+L(u.lane3),True)
print('== C 2連単1点（点数1位→2位、2位と3番手の差）')
for g in (5,7,9):
    s=T[(T.g23>=g)&(T.lane1==1)]; rep(f'①が1位・差{g}以上',s,'nirensho',L(s.lane1)+'-'+L(s.lane2))
print('== D 単勝（点数1位の艇、2番手との差）')
for g in (5,7,9,11):
    s=T[T.g12>=g]; rep(f'差{g}以上 全部',s,'tansho',L(s.lane1)); u=s[s.lane1==1]; rep('  ①',u,'tansho',L(u.lane1)); v=s[s.lane1!=1]; rep('  ①以外',v,'tansho',L(v.lane1))
