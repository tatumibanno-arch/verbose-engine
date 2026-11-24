"""3つの知らせ（t3-lap-one・t3-score-one・t3-score4）の重なり、合わせたときの成績、3連複1点にした場合。オッズは使わない（偶然の比較には確定オッズ）。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
def picks(sub,th):
    Y=X.copy(); Y['sc']=Y[sub].sum(1); Y=Y.sort_values(key+['sc','lane']); Y['o']=Y.groupby(key).cumcount()+1
    T=Y[Y.o<=4].pivot_table(index=key,columns='o',values=['lane','sc','et'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index()
    T=T[(T.lane1==1)&((T.sc4-T.sc3)>=th)]; a,b_=T.lane2.astype(int).astype(str),T.lane3.astype(int).astype(str); f=T.et2<=T.et3
    T['cb']='1-'+np.where(f,a,b_)+'-'+np.where(f,b_,a); T['box']=['-'.join(sorted(['1',x,y])) for x,y in zip(a,b_)]; return T[key+['cb','box']]
S2=picks(['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2'],10).assign(sig='知らせ2')
S3=picks(['r_lap','r_cho','r_ex','r_mo'],5.5).assign(sig='知らせ3')
Lp=X[key+['lane','lap','et']].dropna(subset=['lap']).sort_values(key+['lap','lane']); Lp['o']=Lp.groupby(key).cumcount()+1
W=Lp.pivot_table(index=key,columns='o',values=['lane','lap'],aggfunc='first'); W.columns=[f'{a}{b}' for a,b in W.columns]; W=W.reset_index()
e=X.pivot_table(index=key,columns='lane',values='et').reset_index(); e.columns=key+[f'e{k}' for k in range(1,7)]; W=W.merge(e[key+['e2','e3']],on=key)
W=W[(W.lane1==1)&W[['lane2','lane3']].isin([2,3]).all(1)&((W.lap4-W.lap3)>=0.4)]; W['cb']=np.where(W.e2<=W.e3,'1-2-3','1-3-2'); W['box']='1-2-3'; S1=W[key+['cb','box']].assign(sig='知らせ1')
S=pd.concat([S1,S2,S3]).merge(p,on=key)
f=pd.read_sql("select hd,jcd,rno,combination rf,payout pf from payouts where ticket_type='sanrenfuku' and hd>='20250601'",c).drop_duplicates(key); f['rf']=f.rf.str.replace('=','-').apply(lambda x:'-'.join(sorted(x.split('-')))); S=S.merge(f,on=key,how='left')
S['pay']=np.where(S.res==S.cb,S.payout,0); S['payf']=np.where(S.rf==S.box,S.pf,0)
R=lambda s,col='pay': s[col].sum()/(100*len(s)) if len(s) else np.nan
for sg,s in S.groupby('sig'):
    mo=s.groupby(s.hd.str[:6]).apply(R); print(f'{sg} | レース {len(s)} 3連単1点 {R(s):.3f}（勝ち月 {(mo>1).sum()}/{len(mo)}） 3連複1点 {R(s,"payf"):.3f}（的中 {(s.payf>0).mean():.3f}）')
k=S.groupby(key).sig.apply(lambda x:'+'.join(sorted(x))).value_counts(); print('重なり',k.to_dict())
U=S.drop_duplicates(key+['cb']); mo=U.groupby(U.hd.str[:6]).apply(R)
print(f'3つ合わせて（同じ買い目は1点）| 点数 {len(U)} 1日平均 {len(U)/U.hd.nunique():.2f} 回収 {R(U):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)} 最悪の月 {mo.min():.2f}')
print('月別', mo.round(2).to_dict())
h=U[U.pay>0].pay.sort_values(ascending=False); print('大きい5本を除くと',round((U.pay.sum()-h.head(5).sum())/(100*len(U)),3),' 大きい当たり',list(h.head(5).astype(int)))
# 連敗の長さ
seq=U.sort_values(key).pay.values>0; mx=cur=0
for x in seq: cur=0 if x else cur+1; mx=max(mx,cur)
print('いちばん長い連敗',mx)
print('\n== 100円ずつ買っていたら')
for sg,s in S.groupby('sig'): print(f'{sg}: 購入 {len(s)*100:,}円 払戻 {int(s.pay.sum()):,}円 利益 {int(s.pay.sum()-len(s)*100):,}円')
U=U.sort_values(key); prof=U.pay.values-100; cum=np.cumsum(prof); dd=(cum-np.maximum.accumulate(np.maximum(cum,0))).min()
print(f'3つ合わせて: 購入 {len(U)*100:,}円 払戻 {int(U.pay.sum()):,}円 利益 {int(cum[-1]):,}円 ／ いちばん沈んだとき（それまでの最高からの下げ） {int(dd):,}円')
m=U.groupby(U.hd.str[:6]).apply(lambda s: int(s.pay.sum()-100*len(s))); print('月ごとの利益',m.to_dict())
h=U.pay.sort_values(ascending=False); print('大きい5本を除いた利益',int(U.pay.sum()-h.head(5).sum()-100*len(U)))
print('\n== 大きい当たりの中身')
import sys; sys.path.insert(0,'scripts'); from common import VENUES
top=U.sort_values('pay',ascending=False).head(6)
oo=pd.read_sql("select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd>='20250601'",c).merge(top[key],on=key)
oo['rk']=oo.groupby(key).odds.rank(method='min')
ee=pd.read_sql("select hd,jcd,rno,lane,grade,racer_name from entries where hd>='20250601'",c).merge(top[key],on=key)
for r in top.itertuples():
    q=oo[(oo.hd==r.hd)&(oo.jcd==r.jcd)&(oo.rno==r.rno)&(oo.cb==r.cb)]; g=ee[(ee.hd==r.hd)&(ee.jcd==r.jcd)&(ee.rno==r.rno)].sort_values('lane')
    print(r.hd,VENUES[r.jcd],f'{r.rno}R',r.sig,r.cb,f'{int(r.pay)}円','人気',int(q.rk.iloc[0]) if len(q) else '?','/120',' '.join(f'{l}{gr}' for l,gr in zip(g.lane,g.grade)))
pk=pd.read_sql("select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd>='20250601'",c).merge(U[key+['cb']],on=key+['cb'])
print('知らせの買い目の確定オッズ: 中央値',pk.odds.median(),' 10倍未満',round((pk.odds<10).mean(),3),' 10〜30倍',round(((pk.odds>=10)&(pk.odds<30)).mean(),3),' 30〜100倍',round(((pk.odds>=30)&(pk.odds<100)).mean(),3),' 100倍以上',round((pk.odds>=100).mean(),3))
print('\n== 知らせごと、大きい当たりを除いた利益（100円）')
for sg,s in S.groupby('sig'):
    h=s.pay.sort_values(ascending=False); base=int(s.pay.sum()-100*len(s))
    print(sg,'利益',base,' 大きい1本を除くと',int(base-h.iloc[0]),' 2本',int(base-h.head(2).sum()),' 3本',int(base-h.head(3).sum()),' 大きい当たり',list(h.head(3).astype(int)))
