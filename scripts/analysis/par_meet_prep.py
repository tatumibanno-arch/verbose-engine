"""par_meet_: 節内の勝負駆け・部品交換・体重のための艇ごと枠(2025-06〜)。オッズ不使用。"""
import sqlite3, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/'; key=['hd','jcd','rno']
c=sqlite3.connect('data/boatrace.db')
e=pd.read_sql("select hd,jcd,rno,lane,racer_no,grade,race_name,data from entries where hd>='20250601'",c)
d=pd.DataFrame(e.pop('data').map(json.loads).tolist())
e['sr']=d.series_results.fillna(''); e['nw']=d.national_win_rate; e['mo']=d.motor_2ren_rate
e['racer_no']=e.racer_no.astype(str)
P={'1':10,'2':8,'3':6,'4':4,'5':2,'6':1}
e['pts']=e.sr.map(lambda s: sum(P.get(ch,0) for ch in s)); e['nr']=e.sr.str.len()
e['d']=pd.to_datetime(e.hd)
# 翌日に準優勝戦がある場所 = 予選最終日。準優の出場者数
sf=e[e.race_name.str.contains('準優')].groupby(['jcd','hd']).racer_no.nunique().rename('nsf').reset_index()
sf['d']=pd.to_datetime(sf.hd)-pd.Timedelta(days=1); sf=sf[['jcd','d','nsf']]
e=e.merge(sf,on=['jcd','d'],how='left')
# 準優がある開催でも「前日が準優」の日(=優勝戦日)は除外 → 予選最終日: 当日の race_name に予選を含み翌日準優
e['last']=e.nsf.notna()&e.race_name.str.contains('予選')
# 日頭の分布（各選手の最初の出走のpts）からカットオフ
L=e[e['last']].sort_values(key).drop_duplicates(['hd','jcd','racer_no'])
def cut(g):
    n=int(g.nsf.iloc[0]); v=np.sort(g.pts.values)[::-1]
    return v[min(n,len(v))-1] if len(v) else np.nan
cu=L.groupby(['hd','jcd']).apply(cut).rename('cut').reset_index()
e=e.merge(cu,on=['hd','jcd'],how='left'); e['gap']=e.pts-e.cut
b=pd.read_pickle(S+'bi.pkl'); b['racer_no']=b.racer_no.astype(str)
for q in ('wt','adj','wave_height','wind_speed'): b[q]=pd.to_numeric(b[q],errors='coerce')
X=e.merge(b.drop(columns=['racer_no']),on=key+['lane'],how='left')
r=pd.read_sql("select hd,jcd,rno,lane,position from results where hd>='20250601'",c); r['position']=pd.to_numeric(r.position,errors='coerce')
X=X.merge(r,on=key+['lane'],how='left')
X['ptsd']=X.groupby(key).et.rank(); X.to_pickle(S+'X.pkl')
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250601'",c).drop_duplicates(key); p.to_pickle(S+'p.pkl')
print(len(X), X['last'].mean(), X.gap.describe())
