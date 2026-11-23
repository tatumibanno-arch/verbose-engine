"""単勝①（1号艇）×一周1位が0.935と高かったので掘る。事前に決めた区切りだけ。前半〜2026-04／後半2026-05〜。締切時オッズ（締切前とは違う）。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
o=pd.read_sql("select hd,jcd,rno,lane,lap,mawari,chokusen,lap_kind from oriten",c)
o['lr']=o.groupby(key).lap.rank(method='min'); o['mr']=o.groupby(key).mawari.rank(method='min'); o['cr']=o.groupby(key).chokusen.rank(method='min')
s2=o.sort_values('lap').groupby(key).lap.apply(lambda x:x.iloc[1]-x.iloc[0] if len(x)>1 else np.nan).rename('margin').reset_index()
b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et,json_extract(data,'$.wind_speed') ws,json_extract(data,'$.wave_height') wv from beforeinfo where hd>='20250601'",c); b['er']=b.groupby(key).et.rank(method='min')
for q in ('ws','wv'): b[q]=pd.to_numeric(b[q],errors='coerce')
d=pd.read_sql("select hd,jcd,rno,odds from odds where ticket_type='tansho' and combination='1' and odds>0",c)
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='tansho'",c).drop_duplicates(key)
d=d.merge(p,on=key).merge(o[o.lane==1],on=key).merge(b[b.lane==1][key+['er','ws','wv']],on=key,how='left').merge(s2,on=key,how='left')
d['pay']=np.where(d.res=='1',d.payout,0); cut='20260501'
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s):
    mo=s.groupby(s.hd.str[:6]).apply(R); print(f'{nm} | 点 {len(s)} 的中 {(s.pay>0).mean():.3f} 平均オッズ {s.odds.mean():.2f} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
print('期間',d.hd.min(),d.hd.max()); rep('1号艇 全部',d)
for k in (1,2,3): rep(f'1号艇が一周{k}位',d[d.lr==k])
A=d[d.lr==1]; print('== 1号艇が一周1位のとき')
for lo,hi in ((0,.1),(.1,.2),(.2,.4),(.4,9)): rep(f'  2位との差 {lo}〜{hi}秒',A[(A.margin>=lo)&(A.margin<hi)])
for lo,hi in ((1,1.3),(1.3,1.6),(1.6,2),(2,3),(3,99)): rep(f'  単勝 {lo}〜{hi}倍',A[(A.odds>=lo)&(A.odds<hi)])
rep('  公式展示1位',A[A.er==1]); rep('  公式展示2〜3位',A[A.er.between(2,3)]); rep('  公式展示4〜6位',A[A.er>=4])
rep('  まわり足1位',A[A.mr==1]); rep('  直線1位',A[A.cr==1])
rep('  風0〜3m',A[A.ws<=3]); rep('  風4m以上',A[A.ws>=4]); rep('  波0〜2cm',A[A.wv<=2])
rep('  2位との差0.2秒以上＆風3m以下',A[(A.margin>=.2)&(A.ws<=3)]); rep('  2位との差0.2秒以上＆2倍以上',A[(A.margin>=.2)&(A.odds>=2)])
print('\n== 比較: 一周の順位なしで、単勝オッズ帯だけ（1号艇）')
for lo,hi in ((1,1.6),(1.6,2),(2,3),(3,5),(5,99)): rep(f'  1号艇 単勝{lo}〜{hi}倍（全部）',d[(d.odds>=lo)&(d.odds<hi)])
for k in (1,2,3,4):
    s=d[(d.lr==k if k<4 else d.lr>=4)&(d.odds>=2.5)]; rep(f'  1号艇 一周{k if k<4 else "4〜6"}位 ＆ 2.5倍以上',s)
w=d[d.pay>0]; print('払戻とオッズの一致（的中時）: 払戻/オッズ の中央値',round((w.payout/(w.odds*100)).median(),3),' 平均',round((w.payout/(w.odds*100)).mean(),3))
