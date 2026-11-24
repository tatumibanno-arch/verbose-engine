"""「何艇で決まるか」の度合いを、2艇・4艇に広げる（点数＝7つの順位の合計）。①が点数1位。オッズは使わない。
A 2連複1点（点数上位2艇）・2連単1点（①→点数2位）: 2位と3番手の差 ／ B 4艇で決まる: 4位と5番手の差で、①頭・2・3着を点数2〜4位から（6点）、または3連複（①＋2〜4位から2艇の3点）"""
import numpy as np, pandas as pd, warnings, itertools; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2']; X['sc']=X[cols].sum(1)
Y=X.sort_values(key+['sc','lane']); Y['o']=Y.groupby(key).cumcount()+1
T=Y[Y.o<=5].pivot_table(index=key,columns='o',values=['lane','sc'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index(); T=T[T.lane1==1]
for t in ('sanrensho','sanrenfuku','nirensho','nirenfuku'):
    q=pd.read_sql(f"select hd,jcd,rno,combination r_{t},payout p_{t} from payouts where ticket_type='{t}' and hd>='20250601'",c).drop_duplicates(key); T=T.merge(q,on=key,how='left')
for t in ('sanrenfuku','nirenfuku'): T[f'r_{t}']=T[f'r_{t}'].fillna('').str.replace('=','-').apply(lambda x:'-'.join(sorted(x.split('-'))) if x else '')
T['g23']=T.sc3-T.sc2; T['g45']=T.sc5-T.sc4
L=lambda a: a.astype(int).astype(str)
def rep(nm,s,payfn,npts):
    pay=payfn(s); mo=(pay.groupby(s.hd.str[:6]).sum())/(100*npts*s.groupby(s.hd.str[:6]).size())
    r=lambda m: pay[m].sum()/(100*npts*m.sum()) if m.sum() else np.nan
    print(f'{nm} | レース {len(s)} 的中 {(pay>0).mean():.3f} 回収 {pay.sum()/(100*npts*len(s)):.3f} 前半 {r(s.hd<cut):.3f} 後半 {r(s.hd>=cut):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
print('== A 2艇で決まる（2位と3番手の差）')
for g in (5,7,9,11):
    s=T[T.g23>=g]; pr='-'.join; 
    rep(f'差{g}以上 2連複1点',s,lambda s: pd.Series(np.where(s.r_nirenfuku==['-'.join(sorted(['1',x])) for x in L(s.lane2)],s.p_nirenfuku,0),index=s.index),1)
    rep(f'差{g}以上 2連単1点',s,lambda s: pd.Series(np.where(s.r_nirensho==('1-'+L(s.lane2)),s.p_nirensho,0),index=s.index),1)
print('== B 4艇で決まる（4位と5番手の差）')
def t3six(s):
    out=np.zeros(len(s)); 
    for i,(r,pv,a,b_,c_) in enumerate(zip(s.r_sanrensho,s.p_sanrensho,L(s.lane2),L(s.lane3),L(s.lane4))):
        if isinstance(r,str) and r[0]=='1' and set(r[2::2])<= {a,b_,c_} and len(set(r[2::2]))==2: out[i]=pv
    return pd.Series(out,index=s.index)
def f3three(s):
    out=np.zeros(len(s))
    for i,(r,pv,a,b_,c_) in enumerate(zip(s.r_sanrenfuku,s.p_sanrenfuku,L(s.lane2),L(s.lane3),L(s.lane4))):
        if r and '1' in r.split('-') and set(r.split('-'))-{'1'} <= {a,b_,c_}: out[i]=pv
    return pd.Series(out,index=s.index)
for g in (4,6,8,10):
    s=T[T.g45>=g]; rep(f'差{g}以上 3連単 ①頭・2〜4位で6点',s,t3six,6); rep(f'差{g}以上 3連複 ①＋2〜4位から2艇の3点',s,f3three,3)
