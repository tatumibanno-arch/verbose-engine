"""今の2つの知らせ（t3-lap-one-v1・t3-score-one-v1）の過去の成績が、偶然でどのくらい出るか。
「売れ方どおりに当たりを引き直した世界」（どの買い目も本当は0.75前後）を1000回作り、同じレースで同じ買い目を買ったときの回収率の散らばりと比べる。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts'); import lap_one, score_one
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2']; X['sc']=X[cols].sum(1)
Y=X.sort_values(key+['sc','lane']); Y['o']=Y.groupby(key).cumcount()+1
T=Y[Y.o<=4].pivot_table(index=key,columns='o',values=['lane','sc','et'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index()
T=T[(T.lane1==1)&((T.sc4-T.sc3)>=10)]; L=lambda a: a.astype(int).astype(str)
T['cb']='1-'+np.where(T.et2<=T.et3,L(T.lane2),L(T.lane3))+'-'+np.where(T.et2<=T.et3,L(T.lane3),L(T.lane2)); S1=T[key+['cb']].assign(rule='score')
Lp=X[key+['lane','lap','et']].dropna(subset=['lap']).sort_values(key+['lap','lane']); Lp['o']=Lp.groupby(key).cumcount()+1
W=Lp.pivot_table(index=key,columns='o',values=['lane','lap'],aggfunc='first'); W.columns=[f'{a}{b}' for a,b in W.columns]; W=W.reset_index()
e2=X[X.lane==2][key+['et']].rename(columns={'et':'e2'}); e3=X[X.lane==3][key+['et']].rename(columns={'et':'e3'}); W=W.merge(e2,on=key).merge(e3,on=key)
W=W[(W.lane1==1)&W[['lane2','lane3']].isin([2,3]).all(1)&((W.lap4-W.lap3)>=0.4)]; W['cb']=np.where(W.e2<=W.e3,'1-2-3','1-3-2'); S2=W[key+['cb']].assign(rule='lap')
S=pd.concat([S1,S2]); races=S[key].drop_duplicates()
o=pd.read_sql("select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd>='20250601' and odds>0",c).merge(races,on=key)
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250601'",c).drop_duplicates(key)
S=S.merge(p,on=key).merge(o.rename(columns={'odds':'pick_odds'}),on=key+['cb'],how='left'); S['pay']=np.where(S.res==S.cb,S.payout,0)
o['inv']=1/o.odds; o['rid']=o.hd+o.jcd+o.rno.astype(str); g=o.groupby('rid'); o['q']=o.inv/g.inv.transform('sum')
S['rid']=S.hd+S.jcd+S.rno.astype(str); pq=o.set_index(['rid','cb']).q; S['q']=[pq.get((r,cb),0.0) for r,cb in zip(S.rid,S.cb)]
rng=np.random.default_rng(7)
for rule,s in S.groupby('rule'):
    s=s.dropna(subset=['pick_odds']); real=s.pay.sum()/(100*len(s))
    sims=np.array([((rng.random(len(s))<s.q.values)*s.pick_odds.values).sum()/len(s) for _ in range(2000)])
    print(np.quantile(sims,[.025,.5,.975]).round(4), len(s), s.q.sum().round(2)); print(f"[{rule}] レース {len(s)} 本物の回収 {real:.3f} ｜ 偶然の世界2000回: 平均 {sims.mean():.3f}、95%の範囲 {np.quantile(sims,.025):.3f}〜{np.quantile(sims,.975):.3f}、本物以上 {int((sims>=real).sum())}/2000 ｜ 買い目の売れ方から見た当たる確率の平均 {s.q.mean():.3f}・実際の的中 {(s.pay>0).mean():.3f}")
