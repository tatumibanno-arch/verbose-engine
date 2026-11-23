"""岩郷(慶應・長倉研)卒論「競艇における穴バイアスを利用した舟券購入方法の検討」の追試。
主張: 大村・一般戦で、①の当地勝率6.5以上のレースだけ 1-3 を2連単か2連複（直前オッズの高い方）で買うと回収率100%超
（2018年3〜5月の564レースで発見122%、2017年3〜5月101.8%、2018年10〜12月100.2%）。
ここでは確定払戻で、2連単1-3と2連複1=3を別々に、2005年〜で数える（過去のオッズが無いので「高い方」は選べない）。"""
import sqlite3, pandas as pd, numpy as np, sys
sys.path.insert(0,'scripts'); from common import VENUES
c=sqlite3.connect('data/boatrace.db')
e=pd.read_sql("select hd,jcd,rno,race_name,json_extract(data,'$.local_win_rate') lw from entries where lane=1 and hd>='20050101'",c); e['lw']=pd.to_numeric(e.lw,errors='coerce')
p=pd.read_sql("select hd,jcd,rno,ticket_type t,combination cb,payout from payouts where ticket_type in ('nirensho','nirenfuku') and hd>='20050101'",c)
t=p[p.t=='nirensho'].drop_duplicates(['hd','jcd','rno']).rename(columns={'cb':'r2t','payout':'p2t'})[['hd','jcd','rno','r2t','p2t']]
f=p[p.t=='nirenfuku'].drop_duplicates(['hd','jcd','rno']).rename(columns={'cb':'r2f','payout':'p2f'})[['hd','jcd','rno','r2f','p2f']]
d=e.merge(t,on=['hd','jcd','rno']).merge(f,on=['hd','jcd','rno'],how='left'); d['y']=d.hd.str[:4]
d['t13']=np.where(d.r2t=='1-3',d.p2t,0); d['f13']=np.where(d.r2f.isin(['1-3','1=3']),d.p2f,0); d['best']=d[['t13','f13']].max(axis=1)
print('2連複の表記例',d.r2f.dropna().iloc[0],' 級の手がかり(race_name)例',d.race_name.dropna().unique()[:8])
def rep(nm,s):
    print(f'{nm} | レース {len(s)} 1-3的中 {(s.t13>0).mean():.3f} 2連単 {s.t13.sum()/(100*len(s)):.3f} 2連複 {s.f13.sum()/(100*len(s)):.3f} 両方のうち高い方が必ず取れた場合(上限) {s.best.sum()/(100*len(s)):.3f}')
o=d[(d.jcd=='24')]
print('== 論文と同じ期間（大村）'); 
for a,b in (('20180301','20180531'),('20170301','20170531'),('20181001','20181231')):
    s=o[(o.hd>=a)&(o.hd<=b)]; rep(f'{a}〜{b} 全レース',s); rep(f'  ①当地6.5以上',s[s.lw>=6.5])
print('== 大村・①当地6.5以上、年ごと')
for y,s in o[o.lw>=6.5].groupby('y'): rep(y,s)
rep('大村 2005〜2026 通算',o[o.lw>=6.5]); rep('大村 2019〜2026（論文より後）',o[(o.lw>=6.5)&(o.hd>='20190101')])
print('== 全場・①当地6.5以上'); a=d[d.lw>=6.5]; rep('全場 2005〜2026',a); rep('全場 2019〜2026',a[a.hd>='20190101'])
g=a[a.hd>='20190101'].groupby('jcd').apply(lambda s:pd.Series({'n':len(s),'2連単':round(s.t13.sum()/(100*len(s)),3),'2連複':round(s.f13.sum()/(100*len(s)),3)})); g.index=[VENUES[j] for j in g.index]; print(g.sort_values('2連単',ascending=False).T.to_string())
print('== 大村・当地成績の段階別（2019〜2026、2連単1-3）')
for lo,hi in ((0,5.5),(5.5,6),(6,6.5),(6.5,7),(7,7.5),(7.5,20)): rep(f'{lo}〜{hi}',o[(o.hd>='20190101')&(o.lw>=lo)&(o.lw<hi)])
