"""3連単（締切時100倍未満の組）に、入り残しが見つかった材料を全部足して、学習に使っていない期間で回収率を見る。
材料: 売れ方、1〜3着に置く艇それぞれの オリジナル展示（一周・まわり足・直線の順位）、公式展示順位、モーター差、体重差、風、波。
学習=2025-06〜2026-01、確認=2026-02〜2026-11（学習に使っていない）。締切時オッズ（締切前とは違う）。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'; key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
O=pd.read_parquet(D+'cheap100.parquet'); O=O[O.hd>='20250601']
o=pd.read_sql("select hd,jcd,rno,lane,lap,mawari,chokusen from oriten",c)
for q in ('lap','mawari','chokusen'): o[q]=o.groupby(key)[q].rank(method='min')
b=pd.read_sql("select hd,jcd,rno,lane,json_extract(data,'$.weight') wt,json_extract(data,'$.wind_speed') ws,json_extract(data,'$.wave_height') wv from beforeinfo where hd>='20250601'",c)
for q in ('wt','ws','wv'): b[q]=pd.to_numeric(b[q],errors='coerce')
b['wtd']=b.wt-b.groupby(key).wt.transform('mean'); o=o.merge(b[key+['lane','wtd']],on=key+['lane'],how='left'); env=b.groupby(key)[['ws','wv']].max().reset_index()
O=O.merge(o[key].drop_duplicates(),on=key)
for p in 'abc':
    O=O.merge(o.rename(columns={'lane':f'{p}_lane','lap':f'{p}_lap','mawari':f'{p}_maw','chokusen':f'{p}_cho','wtd':f'{p}_wtd'}),on=key+[f'{p}_lane'],how='left')
O=O.merge(env,on=key,how='left'); O['pay']=np.where(O.hit==1,O.payout,0); O['lq']=np.log(O.q/(1-O.q))
print('組',len(O),'レース',O.groupby(key).ngroups,'期間',O.hd.min(),O.hd.max())
X=['lq','ph','odds']+[f'{p}_{f}' for p in 'abc' for f in ('lap','maw','cho','er','mod','wtd','lane')]+['ws','wv']
tr=O[O.hd<'20260201']; te=O[O.hd>='20260201'].copy(); print('学習',len(tr),'確認',len(te))
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s): 
    mo=s.groupby(s.hd.str[:6]).apply(R); print(f'{nm} | 点 {len(s)} レース {s.groupby(key).ngroups} 的中 {s.hit.mean():.3f} 平均オッズ {s.odds.mean():.1f} 回収 {R(s):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
print('\n=== 手で決めた素直な条件（確認期間）')
w=te[['a_lap','b_lap','c_lap']].max(1); rep('全部',te); rep('3艇とも一周3位以内',te[w<=3]); rep('3艇とも一周3位以内＆20倍未満',te[(w<=3)&(te.odds<20)]); rep('3艇とも一周3位以内＆10倍未満',te[(w<=3)&(te.odds<10)])
rep('1着の艇が一周1位＆2着の艇が一周2位以内＆30倍未満',te[(te.a_lap==1)&(te.b_lap<=2)&(te.odds<30)])
rep('3艇とも一周3位以内＆公式展示も3位以内＆20倍未満',te[(w<=3)&(te[['a_er','b_er','c_er']].max(1)<=3)&(te.odds<20)])
rep('3艇とも一周3位以内＆風4m未満＆20倍未満',te[(w<=3)&(te.ws<4)&(te.odds<20)])
for nm,m in (('ロジスティック',LogisticRegression(C=0.5,max_iter=3000)),('勾配ブースティング',HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=500,l2_regularization=2.0,random_state=0))):
    Xtr=tr[X].fillna(tr[X].median()); Xte=te[X].fillna(tr[X].median()); m.fit(Xtr,tr.hit); te['pr']=m.predict_proba(Xte)[:,1]; te['edge']=te.pr*te.odds
    print(f'\n=== {nm}（確認期間、見立て×オッズの高い順）'); qs=te.edge.quantile([.5,.8,.9,.95,.98,.99]).values; lo=-1
    for lab,hi in zip(['下位50%','50〜80%','80〜90%','90〜95%','95〜98%','98〜99%'],qs): rep('  '+lab,te[(te.edge>lo)&(te.edge<=hi)]); lo=hi
    rep('  上位1%',te[te.edge>lo])
    for th in (1.0,1.1,1.2,1.3): rep(f'  見立て×オッズ{th}以上',te[te.edge>=th])
