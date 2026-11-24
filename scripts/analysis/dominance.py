"""「3艇で決まる度合い」を、一周タイムの差と7つの順位の点数の差の両方で測る（両方そろう／どちらか）。①頭の3連単1点・3連複1点。オッズは使わない。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2']; X['sc']=X[cols].sum(1)
def top(by,asc=True):
    Y=X.sort_values(key+[by,'lane'],ascending=[True,True,True,asc,True]); Y['o']=Y.groupby(key).cumcount()+1
    T=Y[Y.o<=4].pivot_table(index=key,columns='o',values=['lane',by,'et'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; return T.reset_index()
S=top('sc'); S['gs']=S.sc4-S.sc3; Lp=top('lap'); Lp['gl']=Lp.lap4-Lp.lap3
M=S[key+['lane1','lane2','lane3','et2','et3','gs']].merge(Lp[key+['lane1','lane2','lane3','gl']].rename(columns={'lane1':'l1','lane2':'l2','lane3':'l3'}),on=key).merge(p,on=key)
q=pd.read_sql("select hd,jcd,rno,combination rf,payout pf from payouts where ticket_type='sanrenfuku' and hd>='20250601'",c).drop_duplicates(key); M=M.merge(q,on=key,how='left')
M=M[M.lane1==1]; same=(M[['lane1','lane2','lane3']].apply(lambda r:frozenset(r),axis=1)==M[['l1','l2','l3']].apply(lambda r:frozenset(r),axis=1))
a,b_=M.lane2.astype(int).astype(str),M.lane3.astype(int).astype(str); f=M.et2<=M.et3
M['cb']='1-'+np.where(f,a,b_)+'-'+np.where(f,b_,a); M['pay']=np.where(M.res==M.cb,M.payout,0)
M['box']=['-'.join(sorted(['1',x,y])) for x,y in zip(a,b_)]; M['payf']=np.where(M.rf.fillna('').str.replace('=','-').apply(lambda x:'-'.join(sorted(x.split('-'))) if x else '')==M.box,M.pf,0)
def rep(nm,s):
    for col,t in (('pay','3連単1点'),('payf','3連複1点')):
        mo=s.groupby(s.hd.str[:6])[col].sum()/(100*s.groupby(s.hd.str[:6]).size())
        print(f'{nm} {t} | レース {len(s)} 的中 {(s[col]>0).mean():.3f} 回収 {s[col].sum()/(100*len(s)):.3f} 前半 {s[s.hd<cut][col].sum()/(100*max((s.hd<cut).sum(),1)):.3f} 後半 {s[s.hd>=cut][col].sum()/(100*max((s.hd>=cut).sum(),1)):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
print('①が点数1位のレース',len(M),' 点数の上位3艇と一周の上位3艇が同じ',int(same.sum()))
for gs,gl in ((6,0.2),(7,0.2),(7,0.3),(8,0.2),(8,0.3),(6,0.3)):
    rep(f'両方: 点数差{gs}以上＆一周差{gl}秒以上＆同じ3艇',M[same&(M.gs>=gs)&(M.gl>=gl)])
rep('どちらか: 点数差10以上 または（同じ3艇で一周差0.4秒以上）',M[(M.gs>=10)|(same&(M.gl>=.4))])
