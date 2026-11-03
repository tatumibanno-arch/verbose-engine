import sqlite3, numpy as np, pandas as pd
con=sqlite3.connect('data/boatrace.db',timeout=600)
h=pd.read_parquet('data/models/tansho_ev/history.parquet'); h=h[(h.hd>='20251101')&(h.hd<='20261031')]
w=pd.read_sql("select hd,jcd,rno,lane from results where position=1 and hd between '20251101' and '20261031'",con).drop_duplicates(['hd','jcd','rno'])
pt=pd.read_sql("select hd,jcd,rno,payout from payouts where ticket_type='tansho' and hd between '20251101' and '20261031'",con).drop_duplicates(['hd','jcd','rno'])
o=pd.read_sql("select hd,jcd,rno,combination c,odds from odds where ticket_type='tansho' and hd between '20251101' and '20261031'",con)
d=h.merge(w,on=['hd','jcd','rno']).merge(pt,on=['hd','jcd','rno'],how='left')
L=[]
for l in range(1,7):
    x=d[['hd','jcd','rno','lane','payout',f'p1r_{l}']].rename(columns={f'p1r_{l}':'p'}); x['boat']=l; x['win']=x.lane==l; L.append(x)
x=pd.concat(L); o['boat']=o.c.astype(int); x=x.merge(o[['hd','jcd','rno','boat','odds']],on=['hd','jcd','rno','boat'])
x=x[x.odds>0]; s=x.groupby(['hd','jcd','rno']).odds.transform(lambda v:(1/v).sum()); x['pm']=(1/x.odds)/s; x['ratio']=x.p/x.pm
x['ret']=np.where(x.win,x.payout.fillna(0),0); x['m']=x.hd.str[:6]; ndays=x.hd.nunique()
rows=[]
for rmin in [1.6,2.0,2.5,3.0]:
  for omax in [1e9,30,15]:
    for pmin in [0,0.1,0.2]:
      s_=x[(x.ratio>=rmin)&(x.odds<=omax)&(x.p>=pmin)]
      if len(s_)==0: continue
      mm=s_.groupby('m').apply(lambda g:g.ret.sum()/(len(g)*100))
      rows.append(dict(ratio=rmin,odds_max=omax if omax<1e8 else '-',p_min=pmin,per_day=len(s_)/ndays,hit=s_.win.mean(),roi=s_.ret.sum()/(len(s_)*100),win_m=int((mm>1).sum()),worst=mm.min()))
print(pd.DataFrame(rows).round(3).to_string(index=False)); print('days',ndays,'variants',len(rows))
print("\n[A] 当たった艇: 払戻 vs 締切オッズ×100")
wn=x[x.win&x.payout.notna()]; print("一致率(±10円)", (abs(wn.payout-wn.odds*100)<=10).mean().round(3), " 相関", np.corrcoef(wn.payout,wn.odds*100)[0,1].round(4))
# 3連単オッズから艇ごとの1着確率（市場）を作る
o3=pd.read_sql("select hd,jcd,rno,combination c,odds from odds where ticket_type='sanrensho' and hd between '20251101' and '20261031' and odds>0",con)
o3['boat']=o3.c.str[0].astype(int); o3['ip']=1/o3.odds
m3=o3.groupby(['hd','jcd','rno','boat']).ip.sum().reset_index(); m3['pm3']=m3.ip/m3.groupby(['hd','jcd','rno']).ip.transform('sum')
y=x.merge(m3[['hd','jcd','rno','boat','pm3']],on=['hd','jcd','rno','boat'])
ll=lambda p:-np.log(np.clip(p,1e-6,1)); yw=y[y.win]
print("[B] 1着艇の対数損失: モデル",ll(yw.p).mean().round(4)," 単勝市場",ll(yw.pm).mean().round(4)," 3連単市場から",ll(yw.pm3).mean().round(4))
y['r3']=y.p/y.pm3
for col,name in [('ratio','単勝市場とのずれ'),('r3','3連単市場とのずれ')]:
    for rmin in [1.6,2.0,2.5]:
        s_=y[(y[col]>=rmin)&(y.p>=0.1)]; mm=s_.groupby('m').apply(lambda g:g.ret.sum()/(len(g)*100))
        print(f"[C] {name}>={rmin} p>=0.1: 1日{len(s_)/ndays:.1f}点 回収率{s_.ret.sum()/(len(s_)*100):.3f} 勝ち月{(mm>1).sum()}/12")
# 単勝市場 vs 3連単市場: 単勝で過小評価された艇
y['tq']=y.pm3/y.pm
for rmin in [1.5,2.0]:
    s_=y[(y.tq>=rmin)]; print(f"[D] 3連単市場/単勝市場>={rmin}（モデル無し）: 1日{len(s_)/ndays:.1f}点 回収率{s_.ret.sum()/(len(s_)*100):.3f}")
print("\n[E] 3連単市場/単勝市場 のずれ × 3連単市場の1着確率の下限")
for rmin in [1.5,2.0,2.5]:
  for pmin in [0.1,0.2,0.3]:
    s_=y[(y.tq>=rmin)&(y.pm3>=pmin)]; mm=s_.groupby('m').apply(lambda g:g.ret.sum()/(len(g)*100))
    print(f"ずれ>={rmin} 3連単確率>={pmin}: 1日{len(s_)/ndays:.1f}点 的中{s_.win.mean():.2f} 回収率{s_.ret.sum()/(len(s_)*100):.3f} 勝ち月{(mm>1).sum()}/12 最悪月{mm.min():.2f} 平均オッズ{s_.odds.median():.1f}")
