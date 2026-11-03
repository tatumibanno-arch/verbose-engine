import sqlite3, numpy as np, pandas as pd, json, gzip
con=sqlite3.connect('data/boatrace.db')
h=pd.read_parquet('data/models/tansho_ev/history.parquet')
h=h[(h.hd>='20251101')&(h.hd<='20261031')]
w=pd.read_sql("select hd,jcd,rno,lane from results where position=1 and hd between '20251101' and '20261031'",con)
w=w.drop_duplicates(['hd','jcd','rno'])
pt=pd.read_sql("select hd,jcd,rno,combination c,payout from payouts where ticket_type='tansho' and hd between '20251101' and '20261031'",con).drop_duplicates(['hd','jcd','rno'])
o=pd.read_sql("select hd,jcd,rno,combination c,odds from odds where ticket_type='tansho' and hd between '20251101' and '20261031'",con)
d=h.merge(w,on=['hd','jcd','rno']).merge(pt[['hd','jcd','rno','payout']],on=['hd','jcd','rno'],how='left')
L=[]
for l in range(1,7):
    x=d[['hd','jcd','rno',f'p1r_{l}',f'g_{l}','lane','payout']].rename(columns={f'p1r_{l}':'p',f'g_{l}':'g'})
    x['boat']=l; x['win']=(x.lane==l).astype(int); L.append(x)
x=pd.concat(L)
o['boat']=o.c.astype(int); x=x.merge(o[['hd','jcd','rno','boat','odds']],on=['hd','jcd','rno','boat'],how='left')
x['pm']=np.nan
# market prob from closing odds
s=x.groupby(['hd','jcd','rno']).odds.transform(lambda v:(1/v.where(v>0)).sum())
x['pm']=(1/x.odds.where(x.odds>0))/s
print("races",len(d))
print("\n[1] モデル1着確率の当たり具合（2025-11〜2026-10、学習に使っていない月）")
x['bin']=pd.cut(x.p,[0,.05,.1,.2,.3,.4,.5,.6,.7,.8,1])
t=x.groupby('bin',observed=True).agg(n=('win','size'),pred=('p','mean'),act=('win','mean'))
print(t.round(3).to_string())
print("\n[2] 艇番別 予想 vs 実際")
print(x.groupby('boat').agg(pred=('p','mean'),act=('win','mean')).round(3).to_string())
y=x[x.odds.notna()&(x.odds>0)]
ll=lambda p: -np.log(np.clip(p,1e-6,1))
print("\n[3] 当てる力（1着艇の対数損失、低いほど良い / 本命一致率）")
yy=y[y.win==1]
print(f"モデル {ll(yy.p).mean():.3f}  市場(締切オッズ) {ll(yy.pm).mean():.3f}  races={len(yy)}")
top_m=y.loc[y.groupby(['hd','jcd','rno']).p.idxmax() if False else y.sort_values('p').groupby(['hd','jcd','rno']).tail(1).index]
top_m=y.sort_values('p').groupby(['hd','jcd','rno']).tail(1); top_k=y.sort_values('pm').groupby(['hd','jcd','rno']).tail(1)
print(f"本命の1着率 モデル {top_m.win.mean():.3f} 市場 {top_k.win.mean():.3f}")
print("\n[4] 予想配当 g と 実際の単勝配当（当たった艇）")
z=x[(x.win==1)&x.payout.notna()]
z['r']=z.payout/z.g
z['gb']=pd.cut(z.g,[0,130,160,200,300,500,1000,1e5])
print(z.groupby('gb',observed=True).agg(n=('r','size'),g=('g','median'),pay=('payout','median'),ratio=('r','median')).round(2).to_string())
print("\n[5] モデルと市場のずれ: モデル確率/市場確率 の帯ごとの実際の1着率と単勝回収率")
y=y.assign(ratio=y.p/y.pm)
y['rb']=pd.cut(y.ratio,[0,.7,.85,1,1.15,1.3,1.6,100])
y['ret']=np.where(y.win==1,y.payout,0)
print(y.groupby('rb',observed=True).agg(n=('win','size'),model=('p','mean'),market=('pm','mean'),act=('win','mean'),roi=('ret',lambda r:r.sum()/len(r)/100)).round(3).to_string())
print("\n[6] 市場（締切単勝オッズ）の当たり具合")
y['mb']=pd.cut(y.pm,[0,.02,.05,.1,.2,.3,.5,.7,1])
print(y.groupby('mb',observed=True).agg(n=('win','size'),market=('pm','mean'),act=('win','mean'),roi=('ret',lambda r:r.sum()/len(r)/100)).round(3).to_string())
print("\n[7] ずれ≥1.6 の月別・オッズ帯別")
s=y[y.ratio>=1.6].copy(); s['m']=s.hd.str[:6]
print(s.groupby('m').agg(n=('win','size'),roi=('ret',lambda r:r.sum()/len(r)/100)).round(2).T.to_string())
s['ob']=pd.cut(s.odds,[0,5,10,20,50,100,1e4])
print(s.groupby('ob',observed=True).agg(n=('win','size'),act=('win','mean'),roi=('ret',lambda r:r.sum()/len(r)/100)).round(3).to_string())
