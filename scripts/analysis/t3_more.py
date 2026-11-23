"""3連単で、さっきの「①頭×一周」以外の形を探す（運用者「これ以外で」）。事前に決めた形だけ。2025-06〜、前半〜2026-01／後半2026-02〜。締切時オッズ（締切前とは違う）。
A: ①以外の頭（②〜④）が一周1位で、①が一周4〜6位（まくり・差しの形）
B: 売れすぎの艇を外す形を、公式展示ではなく一周タイムで判定
C: ①頭の形に潮を足す（江戸川・徳山・福岡の潮が低い／下関の潮が高い）
D: 2着だけ一周で選ぶ（①頭、2着が一周1〜2位、3着は誰でも）"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
D=ob.D; key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db'); cut='20260201'
O=pd.read_parquet(D+'cheap100.parquet'); O=O[O.hd>='20250601']
o=pd.read_sql("select hd,jcd,rno,lane,lap from oriten",c); o['lap']=o.groupby(key).lap.rank(method='min')
for p in 'abc': O=O.merge(o.rename(columns={'lane':f'{p}_lane','lap':f'{p}_lap'}),on=key+[f'{p}_lane'],how='inner')
l1=o[o.lane==1][key+['lap']].rename(columns={'lap':'lap1'}); O=O.merge(l1,on=key,how='left')
b=pd.read_sql("select hd,jcd,rno,max(cast(json_extract(data,'$.wave_height') as real)) wv,max(cast(json_extract(data,'$.wind_speed') as real)) ws from beforeinfo where hd>='20250601' group by 1,2,3",c); O=O.merge(b,on=key,how='left')
O['pay']=np.where(O.hit==1,O.payout,0)
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s):
    mo=s.groupby(s.hd.str[:6]).apply(R); print(f'{nm} | 点 {len(s)} レース {s.groupby(key).ngroups} 的中 {s.hit.mean():.3f} 平均オッズ {s.odds.mean():.1f} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
import sys as _s
if 'skipA' in _s.argv: pass
print('== A: ①以外の頭が一周1位、①が一周4〜6位')
A=O[(O.a_lane.between(2,4))&(O.a_lap==1)&(O.lap1>=4)]; rep('全部',A); rep('2・3着とも一周3位以内',A[(A.b_lap<=3)&(A.c_lap<=3)]); rep('　＋50倍未満',A[(A.b_lap<=3)&(A.c_lap<=3)&(A.odds<50)])
rep('　＋頭の売れ方30%未満',A[(A.b_lap<=3)&(A.c_lap<=3)&(A.ph<.3)]); rep('　＋波2cm以下',A[(A.b_lap<=3)&(A.c_lap<=3)&(A.wv<=2)])
A2=O[(O.a_lane.between(2,4))&(O.a_lap==1)&(O.lap1<=2)]; rep('（比較）①以外の頭が一周1位、①も一周2位',A2[(A2.b_lap<=3)&(A2.c_lap<=3)])
print('== B: 売れすぎ外しを一周タイムで')
C=pd.read_parquet(D+'ob3.parquet'); C=C[C.hd>='20250601']; C=C.merge(pd.read_parquet(D+'ob2_bets.parquet')[['rno']],left_on='i',right_index=True)
x=o.rename(columns={'lane':'X','lap':'xlap'}); C=C.merge(x,on=key+['X'],how='inner')
def repc(nm,s):
    mo=s.groupby(s.hd.str[:6]).apply(R); print(f'{nm} | 点 {len(s)} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
repc('外す艇の一周5〜6位',C[C.xlap>=5]); repc('　＋売れすぎ幅0.1以上',C[(C.xlap>=5)&(C.gap>=.1)]); repc('　＋売れすぎ幅0.1以上＆15倍未満',C[(C.xlap>=5)&(C.gap>=.1)&(C.odds<15)]); repc('　＋売れすぎ幅0.1以上＆30倍未満',C[(C.xlap>=5)&(C.gap>=.1)&(C.odds<30)])
repc('　＋公式展示も5〜6位＆売れすぎ幅0.1以上',C[(C.xlap>=5)&(C.xer>=5)&(C.gap>=.1)])
print('== C: ①頭の形（①一周1位・2・3着とも一周3位以内）に潮')
T=pd.read_parquet(D+'tide.parquet'); T=T[~T.jcd.isin(['12','13','23'])]
e=pd.read_sql("select distinct hd,jcd,rno,close_time from entries where hd>='20250601'",c); e['hour']=e.close_time.str[:2].astype(float)
T['lv']=T.groupby('jcd')['level'].transform(lambda s:pd.qcut(s,3,labels=['低','中','高']))
Q=O[(O.a_lane==1)&(O.a_lap==1)&(O.b_lap<=3)&(O.c_lap<=3)].merge(e,on=key).merge(T[['jcd','hd','hour','lv']],on=['jcd','hd','hour'])
good=((Q.jcd.isin(['03','18','22']))&(Q.lv=='低'))|((Q.jcd=='19')&(Q.lv=='高'))
rep('潮の条件が合う場面',Q[good]); rep('潮の条件が合わない場面（比較）',Q[~good])
print('== D: ①頭、2着が一周1〜2位（①以外）、3着は誰でも')
Dd=O[(O.a_lane==1)&(O.b_lap<=2)]; rep('全部',Dd); rep('①頭50%未満',Dd[Dd.ph<.5]); rep('①頭50%未満＆3着が一周3位以内',Dd[(Dd.ph<.5)&(Dd.c_lap<=3)]); rep('①一周2〜3位＆2着が一周1位',O[(O.a_lane==1)&(O.a_lap.between(2,3))&(O.b_lap==1)])
print('\n== E: まとめた形「頭＝一周1位の艇（何号艇でも）、2・3着＝一周2・3位の艇」＝1レース2点')
U=O[(O.a_lap==1)&(O.b_lap<=3)&(O.c_lap<=3)]; rep('全部',U)
for lo,hi in ((0,.2),(.2,.35),(.35,.5),(.5,.7),(.7,1.01)): rep(f'  頭の売れ方{int(lo*100)}〜{int(hi*100)}%',U[(U.ph>=lo)&(U.ph<hi)])
V=U[U.ph<.5]; rep('頭の売れ方50%未満',V); rep('　＋波2cm以下',V[V.wv<=2]); rep('　＋風3m以下',V[V.ws<=3]); rep('　＋波2cm以下＆風3m以下',V[(V.wv<=2)&(V.ws<=3)])
rep('　＋頭が1号艇',V[V.a_lane==1]); rep('　＋頭が2〜4号艇',V[V.a_lane.between(2,4)]); rep('　＋頭が5〜6号艇',V[V.a_lane>=5])
