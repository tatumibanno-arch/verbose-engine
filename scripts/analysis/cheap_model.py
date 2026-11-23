"""安い組（締切時20倍未満の3連単）ぜんぶを対象に、売れ方＋艇ごとの材料から「当たる確率」を学習し、
検証期間で見立て÷売れ方の高い順に回収率を見る（条件を機械に総当たりさせる版）。
発見=2025-04〜2026-03、検証=2026-04〜。締切時オッズ（締切前とは違う）。 build / fit"""
import sqlite3, sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'; key=['hd','jcd','rno']
def build():
    c=sqlite3.connect('data/boatrace.db'); rows=[]
    p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250401'",c).drop_duplicates(key)
    for m in pd.period_range('2025-04','2026-11',freq='M'):
        a=m.strftime('%Y%m01'); b=m.strftime('%Y%m31')
        o=pd.read_sql(f"select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd between '{a}' and '{b}' and odds>0",c)
        if o.empty: continue
        o['inv']=1/o.odds; g=o.groupby(key).inv; o['tot']=g.transform('sum'); o['n']=g.transform('size'); o['h']=o.cb.str[0]
        o['ph']=o.groupby(key+['h']).inv.transform('sum')/o.tot; o['rk']=o.groupby(key).odds.rank(method='first')
        rows.append(o[(o.n>=110)&(o.odds<20)].drop(columns=['n','h'])); print(m,flush=True)
    O=pd.concat(rows).merge(p,on=key); O['hit']=(O.cb==O.res).astype(int); O['q']=O.inv/O.tot
    E=pd.read_parquet(D+'ob_ent.parquet')
    for q in ('mo','nw'): E[q]=pd.to_numeric(E[q],errors='coerce')
    E['g']=E.grade.map({'A1':4,'A2':3,'B1':2,'B2':1}); E['c3r']=(E.c3.fillna(0)+3)/(E.cn.fillna(0)+10); E['c1r']=(E.cw.fillna(0)+1)/(E.cn.fillna(0)+10)
    bi=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et,start_st st,start_course sc,json_extract(data,'$.wind_speed') wind from beforeinfo where hd>='20250401'",c)
    for q in ('et','st','wind'): bi[q]=pd.to_numeric(bi[q],errors='coerce')
    bi['etd']=bi.et-bi.groupby(key).et.transform('mean'); bi['sr']=bi.groupby(key).st.rank(method='min'); bi['moved']=(bi.sc!=bi.lane).astype(float)
    e=pd.read_sql("select hd,jcd,rno,lane,racer_no,motor_no from entries where hd>='20250320'",c)
    r=pd.read_sql("select hd,jcd,rno,lane,position from results where hd>='20250320'",c).merge(e,on=key+['lane']); r['pos']=pd.to_numeric(r.position,errors='coerce').fillna(6)
    dd=r[['jcd','hd']].drop_duplicates().sort_values(['jcd','hd']); dd['d']=pd.to_datetime(dd.hd); dd['meet']=(dd.groupby('jcd').d.diff().dt.days!=1).groupby(dd.jcd).cumsum(); r=r.merge(dd[['jcd','hd','meet']],on=['jcd','hd'])
    def prior(k,nm):
        g=r.groupby(['jcd','meet',k,'hd']).pos.agg(['sum','count']).reset_index().sort_values('hd'); g['cs']=g.groupby(['jcd','meet',k])['sum'].cumsum()-g['sum']; g['cc']=g.groupby(['jcd','meet',k])['count'].cumsum()-g['count']
        g[nm]=np.where(g.cc>0,g.cs/g.cc,3.5); return g[['jcd','hd',k,nm]]
    e=e.merge(prior('motor_no','mm'),on=['jcd','hd','motor_no'],how='left').merge(prior('racer_no','rm'),on=['jcd','hd','racer_no'],how='left')
    F=E[key+['lane','g','nw','mo','c3r','c1r','er']].merge(bi[key+['lane','etd','sr','moved','wind']],on=key+['lane'],how='left').merge(e[key+['lane','mm','rm']],on=key+['lane'],how='left')
    for q in ('nw','mo'): F[q+'d']=F[q]-F.groupby(key)[q].transform('mean')
    feats=['g','nwd','mod','c3r','c1r','er','etd','sr','mm','rm']
    for pos,ix in (('a',0),('b',2),('c',4)):
        O['lane']=O.cb.str[ix].astype(int); O=O.merge(F[key+['lane']+feats].rename(columns={f:f'{pos}_{f}' for f in feats}),on=key+['lane'],how='left'); O=O.rename(columns={'lane':f'{pos}_lane'})
    w=F.groupby(key).agg(wind=('wind','max'),moved=('moved','max')).reset_index(); O=O.merge(w,on=key,how='left'); O['jc']=O.jcd.astype(int)
    O.to_parquet(D+'cheap.parquet'); print(len(O),O.hit.mean())
def fit():
    from sklearn.ensemble import HistGradientBoostingClassifier
    O=pd.read_parquet(D+'cheap.parquet'); O['lq']=np.log(O.q/(1-O.q)); O['pay']=np.where(O.hit==1,O.payout,0)
    X=[c for c in O.columns if c[:2] in('a_','b_','c_')]+['lq','ph','rk','odds','wind','moved','rno','jc']
    tr=O[O.hd<'20260101']; es=O[(O.hd>='20260101')&(O.hd<'20260401')]; va=O[O.hd>='20260401'].copy()
    m=HistGradientBoostingClassifier(max_iter=400,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=300,l2_regularization=1.0,early_stopping=False,random_state=0).fit(tr[X],tr.hit)
    R=lambda s: round(s.pay.sum()/(100*len(s)),3) if len(s) else np.nan
    for nm,s in (('確認用 2026-01〜03',es.copy()),('検証 2026-04〜11',va)):
        s['pr']=m.predict_proba(s[X])[:,1]; s['edge']=s.pr*s.odds; print('\n',nm,'全体',len(s),R(s),'的中率',round(s.hit.mean(),4),'見立て平均',round(s.pr.mean(),4))
        qs=s.edge.quantile([.5,.8,.9,.95,.98,.99]).values; lo=-1
        for lab,hi in zip(['下位50%','50〜80%','80〜90%','90〜95%','95〜98%','98〜99%'],qs):
            t=s[(s.edge>lo)&(s.edge<=hi)]; print(f'  見立て×オッズ {lab}: 点 {len(t)} 回収 {R(t)} 平均オッズ {t.odds.mean():.1f}'); lo=hi
        t=s[s.edge>lo]; print(f'  上位1%: 点 {len(t)} レース {t.groupby(key).ngroups} 回収 {R(t)} 平均オッズ {t.odds.mean():.1f}')
        for th in (0.95,1.0,1.05,1.1): t=s[s.edge>=th]; print(f'  見立て×オッズ {th}以上: 点 {len(t)} 回収 {R(t)}')
        if nm.startswith('検証'):
            t=s[s.edge>=s.edge.quantile(.98)]; mo=t.groupby(t.hd.str[:6]).apply(lambda g:pd.Series({'点':len(g),'回収':R(g)})); print('  上位2%の月別'); print(mo.T.to_string()); va=s
    from sklearn.inspection import permutation_importance
    sm=va.sample(60000,random_state=0); pi=permutation_importance(m,sm[X],sm.hit,n_repeats=2,random_state=0,scoring='neg_log_loss'); print('\n効いている材料',[(X[i],round(pi.importances_mean[i],5)) for i in np.argsort(-pi.importances_mean)[:12]])
if __name__=='__main__': build() if sys.argv[1]=='build' else fit()
