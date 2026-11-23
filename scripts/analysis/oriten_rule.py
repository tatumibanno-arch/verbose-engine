"""「3艇とも一周3位以内」をオッズ帯・①頭の売れ方・場で分け、さらに売れすぎ外し（外す艇の一周が5〜6位）と組み合わせる。
2025-06〜2026-11、前半=〜2026-01、後半=2026-02〜。締切時オッズ（締切前とは違う）。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts'); from common import VENUES
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'; key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
O=pd.read_parquet(D+'cheap100.parquet'); O=O[O.hd>='20250601']
o=pd.read_sql("select hd,jcd,rno,lane,lap from oriten",c); o['lap']=o.groupby(key).lap.rank(method='min')
for p in 'abc': O=O.merge(o.rename(columns={'lane':f'{p}_lane','lap':f'{p}_lap'}),on=key+[f'{p}_lane'],how='inner')
O['pay']=np.where(O.hit==1,O.payout,0); O['w']=O[['a_lap','b_lap','c_lap']].max(1)
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s):
    mo=s.groupby(s.hd.str[:6]).apply(R); print(f'{nm} | 点 {len(s)} 回収 {R(s):.3f} 前半 {R(s[s.hd<"20260201"]):.3f} 後半 {R(s[s.hd>="20260201"]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
G=O[O.w<=3]; print('全体',round(R(O),3),' 3艇とも一周3位以内',round(R(G),3))
print('== オッズ帯（3艇とも一周3位以内）')
for lo,hi in ((0,6),(6,10),(10,20),(20,30),(30,50),(50,100)): rep(f'{lo}〜{hi}倍',G[(G.odds>=lo)&(G.odds<hi)])
print('== その頭の売れ方'); 
for lo,hi in ((0,.3),(.3,.5),(.5,.7),(.7,1.01)): rep(f'{int(lo*100)}〜{int(hi*100)}%',G[(G.ph>=lo)&(G.ph<hi)])
print('== 頭の艇の一周順位'); 
for k in (1,2,3): rep(f'頭が一周{k}位',G[G.a_lap==k])
print('== 3艇とも一周2位以内…は3艇入らないので、一周1〜3位の3艇ちょうどの6点（ボックス）'); 
B=G.copy(); rep('一周上位3艇のボックス（各レース6点まで）',B)
print('== 場ごと（3艇とも一周3位以内）'); g=G.groupby('jcd').apply(lambda s:pd.Series({'点':len(s),'回収':round(R(s),3),'前半':round(R(s[s.hd<'20260201']),3),'後半':round(R(s[s.hd>='20260201']),3)})); g.index=[VENUES[j] for j in g.index]; print(g.sort_values('回収',ascending=False).T.to_string())
gd=G[G.hd<'20260201'].groupby('jcd').apply(R); good=gd[gd>=gd.median()].index; print('前半で上半分の場→後半',round(R(G[(G.hd>='20260201')&G.jcd.isin(good)]),3),' 下半分の場→後半',round(R(G[(G.hd>='20260201')&~G.jcd.isin(good)]),3))
