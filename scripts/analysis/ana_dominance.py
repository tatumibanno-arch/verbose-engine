"""穴側でも「3艇で決まる度合い」が効くか。①が点数（7つの順位の合計）で4位以下＝①が飛びそうで、上位3艇が4番手より抜けているレース。オッズは使わない。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2']; X['sc']=X[cols].sum(1)
Y=X.sort_values(key+['sc','lane']); Y['o']=Y.groupby(key).cumcount()+1
T=Y[Y.o<=4].pivot_table(index=key,columns='o',values=['lane','sc','et'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index()
r1=Y[Y.lane==1][key+['o']].rename(columns={'o':'o1'}); T=T.merge(r1,on=key)
for t in ('sanrensho','sanrenfuku'):
    q=pd.read_sql(f"select hd,jcd,rno,combination r_{t},payout p_{t} from payouts where ticket_type='{t}' and hd>='20250601'",c).drop_duplicates(key); T=T.merge(q,on=key,how='left')
T['r_sanrenfuku']=T.r_sanrenfuku.fillna('').str.replace('=','-').apply(lambda x:'-'.join(sorted(x.split('-'))) if x else '')
T['g34']=T.sc4-T.sc3; L=lambda a: a.astype(int).astype(str)
T['t3']=L(T.lane1)+'-'+np.where(T.et2<=T.et3,L(T.lane2),L(T.lane3))+'-'+np.where(T.et2<=T.et3,L(T.lane3),L(T.lane2))
T['box']=['-'.join(sorted([a,b,c_])) for a,b,c_ in zip(L(T.lane1),L(T.lane2),L(T.lane3))]
T['pay3']=np.where(T.r_sanrensho==T.t3,T.p_sanrensho,0); T['payf']=np.where(T.r_sanrenfuku==T.box,T.p_sanrenfuku,0)
def rep(nm,s):
    for col,t in (('pay3','3連単1点'),('payf','3連複1点')):
        mo=s.groupby(s.hd.str[:6])[col].sum()/(100*s.groupby(s.hd.str[:6]).size())
        print(f'{nm} {t} | レース {len(s)} 的中 {(s[col]>0).mean():.3f} 当たり平均 {s[s[col]>0][col].mean():.0f}円 回収 {s[col].sum()/(100*len(s)):.3f} 前半 {s[s.hd<cut][col].sum()/(100*max((s.hd<cut).sum(),1)):.3f} 後半 {s[s.hd>=cut][col].sum()/(100*max((s.hd>=cut).sum(),1)):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
for g in (0,4,6,8,10):
    rep(f'①が点数4位以下・差{g}以上',T[(T.o1>=4)&(T.g34>=g)])
rep('（比較）①が点数2〜3位・差6以上',T[(T.o1.between(2,3))&(T.g34>=6)])
