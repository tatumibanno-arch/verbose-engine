"""単勝で見えた「一周1位の1号艇が売れていないと割安」を3連単に持ち込む。①頭の売れ方（3連単の①頭の合計）と①の一周順位で分け、
①頭の組を、2・3着の一周順位で絞る。2025-06〜2026-11、前半〜2026-01／後半2026-02〜。締切時オッズ（締切前とは違う）。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'; key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
O=pd.read_parquet(D+'cheap100.parquet'); O=O[(O.hd>='20250601')&(O.a_lane==1)]
o=pd.read_sql("select hd,jcd,rno,lane,lap from oriten",c); o['lap']=o.groupby(key).lap.rank(method='min')
for p in 'abc': O=O.merge(o.rename(columns={'lane':f'{p}_lane','lap':f'{p}_lap'}),on=key+[f'{p}_lane'],how='inner')
O['pay']=np.where(O.hit==1,O.payout,0); cut='20260201'
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s):
    mo=s.groupby(s.hd.str[:6]).apply(R); print(f'{nm} | 点 {len(s)} レース {s.groupby(key).ngroups} 的中 {s.hit.mean():.3f} 平均オッズ {s.odds.mean():.1f} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
print('== ①頭の組（100倍未満）を、①頭の売れ方×①の一周順位で')
for lo,hi in ((0,.25),(.25,.35),(.35,.5),(.5,.7),(.7,1.01)):
    s=O[(O.ph>=lo)&(O.ph<hi)]; rep(f'①頭{int(lo*100)}〜{int(hi*100)}% ①一周1位',s[s.a_lap==1]); rep(f'①頭{int(lo*100)}〜{int(hi*100)}% ①一周4〜6位',s[s.a_lap>=4])
print('== ①頭35%未満 ＆ ①一周1位 を、2・3着の一周順位とオッズで')
S=O[(O.ph<.35)&(O.a_lap==1)]
rep('全部',S); rep('2・3着とも一周3位以内',S[(S.b_lap<=3)&(S.c_lap<=3)]); rep('2着が一周3位以内',S[S.b_lap<=3]); rep('2・3着とも一周4位以内',S[(S.b_lap<=4)&(S.c_lap<=4)])
for lo,hi in ((0,15),(15,30),(30,50),(50,100)): rep(f'  2・3着とも一周3位以内＆{lo}〜{hi}倍',S[(S.b_lap<=3)&(S.c_lap<=3)&(S.odds>=lo)&(S.odds<hi)])
S2=O[(O.ph<.5)&(O.a_lap==1)]; print('== ①頭50%未満 ＆ ①一周1位'); rep('全部',S2); rep('2・3着とも一周3位以内',S2[(S2.b_lap<=3)&(S2.c_lap<=3)])
print('\n== 事前に決めた組み合わせ（①頭の組、①一周1位、2・3着とも一周3位以内）にさらに1つ足す')
B=O[(O.a_lap==1)&(O.b_lap<=3)&(O.c_lap<=3)]
b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et,json_extract(data,'$.wind_speed') ws,json_extract(data,'$.wave_height') wv from beforeinfo where hd>='20250601'",c); b['er']=b.groupby(key).et.rank(method='min')
for q in ('ws','wv'): b[q]=pd.to_numeric(b[q],errors='coerce')
env=b.groupby(key)[['ws','wv']].max().reset_index(); B=B.merge(env,on=key,how='left').merge(b[b.lane==1][key+['er']].rename(columns={'er':'er1'}),on=key,how='left')
rep('土台（①頭の売れ方は問わない）',B); rep('①頭50%未満',B[B.ph<.5]); rep('①頭50%未満＆風3m以下',B[(B.ph<.5)&(B.ws<=3)]); rep('①頭50%未満＆波2cm以下',B[(B.ph<.5)&(B.wv<=2)])
rep('①頭50%未満＆①公式展示1〜2位',B[(B.ph<.5)&(B.er1<=2)]); rep('①頭50%未満＆30倍以上',B[(B.ph<.5)&(B.odds>=30)]); rep('①頭40〜60%',B[(B.ph>=.4)&(B.ph<.6)])
