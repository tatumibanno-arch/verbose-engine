"""①頭のレースで「相手の売れすぎ」を、枠別成績・モーターと売れ方の差で測って検証する。
注意: オッズは締切時。締切前のオッズとは違う。 使い方: build / report"""
import sqlite3, sys, json, itertools, numpy as np, pandas as pd
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
def build():
    c=sqlite3.connect('data/boatrace.db'); rows=[]
    for m in pd.period_range('2025-04','2026-11',freq='M'):
        a=m.strftime('%Y%m01'); b=m.strftime('%Y%m31')
        o=pd.read_sql(f"select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd between '{a}' and '{b}' and odds>0",c)
        if o.empty: continue
        o['inv']=1/o.odds; tot=o.groupby(['hd','jcd','rno']).inv.agg(tot='sum',n='size')
        h=o[o.cb.str[0]=='1'].copy(); h['key']='o'+h.cb.str[2]+h.cb.str[4]
        w=h.pivot_table(index=['hd','jcd','rno'],columns='key',values='odds'); rows.append(tot.join(w,how='inner').reset_index()); print(m,flush=True)
    R=pd.concat(rows); R=R[R.n>=110].dropna()
    p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250401'",c).drop_duplicates(['hd','jcd','rno'])
    R=R.merge(p,on=['hd','jcd','rno']); R.to_parquet(D+'ob_races.parquet')
    e=pd.read_sql("select hd,jcd,rno,lane,racer_no,grade,json_extract(data,'$.national_win_rate') nw,json_extract(data,'$.motor_2ren_rate') mo from entries where hd>='20250401'",c)
    s=pd.read_sql("select hd shd,racer_no,data from racer_stats where hd>='20240401'",c); j=s.data.apply(json.loads)
    for k in range(1,7):
        st=j.apply(lambda x:x.get(f'c{k}_starts') or 0)
        for q in (1,2,3): s[f'c{k}p{q}']=j.apply(lambda x:x.get(f'c{k}_pos{q}') or 0)
        s[f'c{k}n']=st
    s=s.drop(columns='data'); e['d']=pd.to_datetime(e.hd); s['d']=pd.to_datetime(s.shd)
    e=pd.merge_asof(e.sort_values('d'),s.sort_values('d'),on='d',by='racer_no',allow_exact_matches=False)
    for q,nm in ((1,'c1'),(2,'c2'),(3,'c3')): pass
    e['cn']=[getattr(r,f'c{r.lane}n') for r in e.itertuples()]
    e['cw']=[getattr(r,f'c{r.lane}p1') for r in e.itertuples()]; e['c2']=e.cw+[getattr(r,f'c{r.lane}p2') for r in e.itertuples()]; e['c3']=e.c2+[getattr(r,f'c{r.lane}p3') for r in e.itertuples()]
    b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et from beforeinfo where hd>='20250401'",c); b['et']=pd.to_numeric(b.et,errors='coerce'); b['er']=b.groupby(['hd','jcd','rno']).et.rank(method='min')
    e=e.merge(b[['hd','jcd','rno','lane','er']],on=['hd','jcd','rno','lane'],how='left')
    e[['hd','jcd','rno','lane','grade','nw','mo','cn','cw','c2','c3','er']].to_parquet(D+'ob_ent.parquet'); print(len(R),len(e))
def load():
    R=pd.read_parquet(D+'ob_races.parquet'); E=pd.read_parquet(D+'ob_ent.parquet')
    oc=[c for c in R.columns if c[0]=='o' and len(c)==3]
    inv=1/R[oc]; R['p1']=inv.sum(1)/R.tot
    L=[]
    for k in range(2,7):
        cols=[c for c in oc if str(k) in c[1:]]; c2=[c for c in oc if c[1]==str(k)]
        x=R[['hd','jcd','rno','res','payout','p1']].copy(); x['lane']=k; x['mk3']=inv[cols].sum(1)/inv.sum(1); x['mk2']=inv[c2].sum(1)/inv.sum(1); L.append(x)
    L=pd.concat(L).merge(E,on=['hd','jcd','rno','lane'],how='left')
    L['win1']=L.res.str[0]=='1'; L['in3']=L.apply(lambda r:str(r.lane) in r.res[2:],axis=1); L['in2']=L.res.str[2]==L.lane.astype(str)
    for c in ('nw','mo'): L[c]=pd.to_numeric(L[c],errors='coerce')
    # 枠別3連率・2連率（少ない出走は場の平均に寄せる）
    base3=L[L.win1].groupby('lane').in3.mean(); base2=L[L.win1].groupby('lane').in2.mean()
    L['cn']=L.cn.fillna(0); L['r3']=(L.c3.fillna(0)+10*L.lane.map(base3))/(L.cn+10); L['r2']=(L.c2.fillna(0)+10*L.lane.map(base2))/(L.cn+10)
    return R,L,oc
def report():
    R,L,oc=load(); L=L.dropna(subset=['mo','nw'])
    from sklearn.linear_model import LogisticRegression
    L['A']=L.grade.isin(['A1','A2']).astype(int)
    X=lambda d:np.column_stack([d.r3,d.r2,d.mo/100,d.nw/10]+[(d.lane==k).astype(int) for k in range(3,7)])
    disc=L[(L.hd<'20260401')&L.win1]; m=LogisticRegression(C=10,max_iter=500).fit(X(disc),disc.in3)
    L['f3']=m.predict_proba(X(L))[:,1]; L['gap']=L.mk3-L.f3
    print('係数 r3,r2,motor,nw:',m.coef_[0][:4].round(2))
    W=L[L.win1&(L.p1>=.4)]
    for nm,s in (('発見 25/4-26/3',W[W.hd<'20260401']),('検証 26/4-11',W[W.hd>='20260401'])):
        g=s.groupby(pd.cut(s.gap,[-1,-.2,-.1,-.05,0,.05,.1,.2,1]),observed=True).agg(n=('in3','size'),売れ方=('mk3','mean'),成績からの見立て=('f3','mean'),実際=('in3','mean')); g['実際−売れ方']=g.実際-g.売れ方
        print('\n①が勝ったレースで、相手1艇ごと（売れ方−成績の見立て）別:',nm); print(g.round(3).to_string())
    # モーターが悪いのに売れている
    W2=W.copy(); W2['mrank']=W2.groupby(['hd','jcd','rno']).mo.rank(ascending=False); W2['krank']=W2.groupby(['hd','jcd','rno']).mk3.rank(ascending=False)
    g=W2[W2.krank==1].groupby('mrank').agg(n=('in3','size'),売れ方=('mk3','mean'),実際=('in3','mean')); g['差']=g.実際-g.売れ方; print('\nいちばん売れている相手の、モーター順位（相手5艇中）別'); print(g.round(3).to_string())
    g=W2[W2.krank>=4].groupby('mrank').agg(n=('in3','size'),売れ方=('mk3','mean'),実際=('in3','mean')); g['差']=g.実際-g.売れ方; print('\n売れていない相手（4〜5番手）の、モーター順位別'); print(g.round(3).to_string())
    # 買い方: いちばん売れすぎの相手Xを外し、残り4艇から成績の見立て上位3艇で 1-abc-abc の6点均等
    L['grk']=L.groupby(['hd','jcd','rno']).gap.rank(ascending=False,method='first')
    rows=[]
    for (hd,j,r),g in L[L.p1>=.4].groupby(['hd','jcd','rno']):
        if len(g)<5: continue
        x=g[g.grk==1].iloc[0]; rest=g[g.grk!=1].sort_values('f3',ascending=False).head(3); ls=sorted(rest.lane)
        rows.append((hd,j,r,x.lane,x.gap,x.mo-rest.mo.mean(),x.r3-0,rest.r3.sum(),x.mk3,x.p1,''.join(map(str,ls))))
    B=pd.DataFrame(rows,columns=['hd','jcd','rno','X','gap','modiff','xr3','restr3','xmk','p1x','ls']).merge(R,on=['hd','jcd','rno'])
    def six(r): return [f"1-{a}-{b}" for a,b in itertools.permutations(r.ls,2)]
    B['inv6']=[sum(1/r[f'o{a}{b}'] for a,b in itertools.permutations(r['ls'],2)) for _,r in B.iterrows()]
    B['syn']=1/B.inv6; B['hit']=[r.res in six(r) for r in B.itertuples()]; B['pay']=np.where(B.hit,B.payout,0); B['mk6']=B.inv6/B.tot
    B.to_parquet(D+'ob_bets.parquet')
    def t(nm,s):
        o=[nm,'n',len(s),'的中',round(s.hit.mean(),3),'売れ方',round(s.mk6.mean(),3),'合成',round(s.syn.median(),1),'回収',round(s.pay.sum()/(600*len(s)),3)]
        for h,g in (('発見',s[s.hd<'20260401']),('検証',s[s.hd>='20260401'])): o+=[h,len(g),round(g.pay.sum()/(600*max(len(g),1)),3)]
        print(*o)
    print('\n買い方: 売れすぎの相手Xを外して 1-(残り上位3艇)-(同) 6点均等  ①頭の売れ方40%以上')
    t('全部',B)
    for a,b in ((0,.05),(.05,.1),(.1,.15),(.15,.2),(.2,1)): t(f'売れすぎ幅 {a}〜{b}',B[(B.gap>=a)&(B.gap<b)])
    t('売れすぎ幅0.1以上 & Xのモーターが残り平均より5以上低い',B[(B.gap>=.1)&(B.modiff<=-5)])
    t('売れすぎ幅0.1以上 & Xのモーターが10以上低い',B[(B.gap>=.1)&(B.modiff<=-10)])
    t('上 & 合成6〜15',B[(B.gap>=.1)&(B.modiff<=-5)&(B.syn>=6)&(B.syn<15)])
    t('上 & ①頭60%以上',B[(B.gap>=.1)&(B.modiff<=-5)&(B.p1x>=.6)])
    t('Xが②',B[(B.gap>=.1)&(B.X==2)]); t('Xが③',B[(B.gap>=.1)&(B.X==3)]); t('Xが④以降',B[(B.gap>=.1)&(B.X>=4)])
    x=B[(B.hd=='20261122')&(B.jcd=='13')&(B.rno==10)]; print(x[['X','gap','modiff','ls','syn','res','payout']])
if __name__=='__main__':
    build() if sys.argv[1]=='build' else report()
