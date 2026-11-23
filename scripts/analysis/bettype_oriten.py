"""券種ごとに、一周タイム（オリジナル展示）と人気側の組み合わせで回収率を見る。単勝・2連単・2連複・3連単。
期間はオッズとオリジナル展示が両方ある範囲。前半/後半で分ける。締切時オッズ（締切前とは違う）。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
o=pd.read_sql("select hd,jcd,rno,lane,lap,mawari,chokusen from oriten",c)
for q in ('lap','mawari','chokusen'): o[q]=o.groupby(key)[q].rank(method='min')
b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et,json_extract(data,'$.wind_speed') ws from beforeinfo where hd>='20250601'",c); b['er']=b.groupby(key).et.rank(method='min'); b['ws']=pd.to_numeric(b.ws,errors='coerce'); o=o.merge(b[key+['lane','er','ws']],on=key+['lane'],how='left')
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s,cut):
    mo=s.groupby(s.hd.str[:6]).apply(R); print(f'  {nm} | 点 {len(s)} 的中 {(s.pay>0).mean():.3f} 平均オッズ {s.odds.mean():.1f} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
def load(t):
    d=pd.read_sql(f"select hd,jcd,rno,combination cb,odds from odds where ticket_type='{t}' and odds>0 and hd>='20250601'",c)
    p=pd.read_sql(f"select hd,jcd,rno,combination res,payout from payouts where ticket_type='{t}' and hd>='20250601'",c)
    return d,p
# 単勝
d,p=load('tansho'); d['lane']=d.cb.astype(int); p=p.drop_duplicates(key); d=d.merge(p,on=key); d['pay']=np.where(d.cb==d.res,d.payout,0); d['rk']=d.groupby(key).odds.rank(method='first')
d=d.merge(o,on=key+['lane']); cut=sorted(d.hd.unique())[len(d.hd.unique())//2]; print('== 単勝',d.hd.min(),d.hd.max(),'境',cut)
rep('全部',d,cut); rep('一周1位',d[d.lap==1],cut); rep('一周1位＆1番人気',d[(d.lap==1)&(d.rk==1)],cut); rep('一周1位＆2〜3番人気',d[(d.lap==1)&d.rk.between(2,3)],cut); rep('一周1位＆4番人気以下',d[(d.lap==1)&(d.rk>=4)],cut)
rep('一周1位＆直線1位',d[(d.lap==1)&(d.chokusen==1)],cut); rep('一周1位＆公式展示1位',d[(d.lap==1)&(d.er==1)],cut); rep('一周1位＆1号艇',d[(d.lap==1)&(d.lane==1)],cut); rep('一周1位＆1号艇＆風4m未満',d[(d.lap==1)&(d.lane==1)&(d.ws<4)],cut)
rep('1号艇＆一周1位＆まわり足1位',d[(d.lane==1)&(d.lap==1)&(d.mawari==1)],cut); rep('1号艇＆一周4〜6位（比較）',d[(d.lane==1)&(d.lap>=4)],cut)
# 2連単・2連複
for t,sep in (('nirensho','-'),('nirenfuku','=')):
    d,p=load(t); p=p.drop_duplicates(key); d=d.merge(p,on=key); d['cbn']=d.cb.str.replace('=','-'); d['resn']=d.res.str.replace('=','-')
    if t=='nirenfuku': d['cbn']=d.cbn.apply(lambda x:'-'.join(sorted(x.split('-')))); d['resn']=d.resn.apply(lambda x:'-'.join(sorted(x.split('-'))))
    d['pay']=np.where(d.cbn==d.resn,d.payout,0); d['a']=d.cbn.str[0].astype(int); d['b']=d.cbn.str[2].astype(int)
    d=d.merge(o[key+['lane','lap','er']].rename(columns={'lane':'a','lap':'la','er':'ea'}),on=key+['a']).merge(o[key+['lane','lap','er']].rename(columns={'lane':'b','lap':'lb','er':'eb'}),on=key+['b'])
    cut=sorted(d.hd.unique())[len(d.hd.unique())//2]; print(f'\n== {"2連単" if t=="nirensho" else "2連複"}',d.hd.min(),d.hd.max(),'境',cut)
    rep('全部',d,cut); g=d[(d.la<=2)&(d.lb<=2)]; rep('2艇とも一周2位以内',g,cut)
    for lo,hi in ((0,5),(5,10),(10,20),(20,50)): rep(f'  そのうち{lo}〜{hi}倍',g[(g.odds>=lo)&(g.odds<hi)],cut)
    g3=d[(d.la<=3)&(d.lb<=3)]; rep('2艇とも一周3位以内',g3,cut); rep('2艇とも一周3位以内＆10倍未満',g3[g3.odds<10],cut)
    if t=='nirensho': rep('頭が一周1位＆2着が一周2位以内',d[(d.la==1)&(d.lb<=2)],cut); rep('頭が一周1位＆1号艇＆2着が一周3位以内',d[(d.la==1)&(d.a==1)&(d.lb<=3)],cut)
    rep('2艇とも一周2位以内＆公式展示も2位以内',d[(d.la<=2)&(d.lb<=2)&(d.ea<=2)&(d.eb<=2)],cut)
