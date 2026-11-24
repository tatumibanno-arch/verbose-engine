"""par_ml_trifecta: 段階ごとの着順モデル(LightGBM)で120通りの確率 × 群衆が見える情報だけの払戻予測 → 期待値で買う。
オッズは特徴量に使わない。払戻は目的変数と評価のみ。学習 2025-06〜2026-01、検証 2026-02〜。"""
import sys, itertools, numpy as np, pandas as pd, warnings, lightgbm as lgb; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
WALK = len(sys.argv)>1 and sys.argv[1]=='walk'
ent=pd.read_sql("select hd,jcd,rno,lane,grade from entries where hd>='20250601'",c)
gm={'A1':3,'A2':2,'B1':1,'B2':0}; ent['g']=ent.grade.map(gm)
X=X.merge(ent[key+['lane','g']],on=key+['lane'],how='left')
X=X[X.groupby(key).lane.transform('count')==6].copy()
X['jc']=X.jcd.astype(int)
base=['nw','mo','c2r','et','st','lap','mawari','chokusen']
for q in base: X[q+'_d']=X[q]-X.groupby(key)[q].transform('mean')
X['nw_max_o']=X.nw-X.groupby(key).nw.transform(lambda s:s.nlargest(2).iloc[-1] if s.notna().sum()>1 else np.nan)
CROWD=['lane','jc','g','nw','mo','c2r','et','nw_d','mo_d','c2r_d','et_d','r_nw','r_mo','r_c2','r_ex','nw_max_o']
FULL=CROWD+['st','lap','mawari','chokusen','st_d','lap_d','mawari_d','chokusen_d','r_lap','r_maw','r_cho','r_st']
X=X.sort_values(key+['lane']).reset_index(drop=True)
rid=X.groupby(key,sort=False).ngroup().values; nR=rid.max()+1
R_=X[key].drop_duplicates().reset_index(drop=True); R_=R_.merge(p,on=key,how='left')
ok=R_.res.notna().values
sp=R_.res.fillna('0-0-0').str.split('-',expand=True).astype(int).values  # lane numbers
first=np.zeros(nR,int);sec=np.zeros(nR,int);thr=np.zeros(nR,int);first[:]=sp[:,0];sec[:]=sp[:,1];thr[:]=sp[:,2]
ln=X.lane.values
X['y1']=(ln==first[rid]).astype(int); X['y2']=(ln==sec[rid]).astype(int); X['y3']=(ln==thr[rid]).astype(int)
X['ok']=ok[rid]
X['mon']=X.hd.str[:6]; Rmon=R_.hd.str[:6].values; Rhd=R_.hd.values
P=dict(objective='binary',learning_rate=0.05,num_leaves=15,min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,verbose=-1,num_threads=4)
NR=250
def fit(df,feats,y): return lgb.train(P,lgb.Dataset(df[feats],df[y]),NR)
combos=np.array(list(itertools.permutations(range(6),3)))  # lane idx
def stage_scores(models,df,feats): return [m.predict(df[feats]).reshape(-1,6) for m in models]
def combo_probs(S):
    s1,s2,s3=S; s1=s1/s1.sum(1,keepdims=True)
    a,b,c=combos.T; n=len(s1)
    p1=s1[:,a]
    d2=s2.sum(1,keepdims=True)-s2[:,a]; p2=s2[:,b]/d2
    d3=s3.sum(1,keepdims=True)-s3[:,a]-s3[:,b]; p3=s3[:,c]/d3
    return p1*p2*p3   # (nR,120)
def train_models(tr,feats):
    m1=fit(tr,feats,'y1')
    t2=tr[tr.y1==0]; m2=fit(t2,feats,'y2')
    t3=tr[(tr.y1==0)&(tr.y2==0)]; m3=fit(t3,feats,'y3')
    return [m1,m2,m3]
def race_frame(df): # df ordered by race, 6 rows each
    return df
def pay_feats(df,pc):  # per race (n,120) combos features
    n=len(df)//6
    V={q:df[q].values.reshape(n,6) for q in ['nw','mo','c2r','et','g','nw_d','mo_d','et_d','r_nw','r_ex']}
    jc=df.jc.values.reshape(n,6)[:,0]
    cols={}
    for pos,idx in enumerate(combos.T):
        for q,v in V.items(): cols[f'{q}{pos}']=v[:,idx].ravel() if False else None
    out=[]
    F={}
    for pos in range(3):
        idx=combos[:,pos]
        F[f'lane{pos}']=np.tile(idx+1,n)
        for q,v in V.items(): F[f'{q}{pos}']=v[:,idx].ravel()
    F['jc']=np.repeat(jc,120); F['lpc']=np.log(pc.ravel()+1e-9)
    F['nwmax']=np.repeat(V['nw'].max(1),120); F['nwmean']=np.repeat(V['nw'].mean(1),120)
    return pd.DataFrame(F)
def run_period(trmask_races,temask_races):
    """returns DataFrame of per-combo test rows: race idx, combo idx, pm, pred pay"""
    Xtr=X[trmask_races[rid]&X.ok.values]; Xte=X[temask_races[rid]]
    mc=train_models(Xtr,CROWD); mf=train_models(Xtr,FULL)
    Pf_te=combo_probs(stage_scores(mf,Xte,FULL))
    # crowd probs for payout feature: for train use in-sample crowd (slightly optimistic but only a feature of price); fine
    Pc_tr=combo_probs(stage_scores(mc,Xtr,CROWD)); Pc_te=combo_probs(stage_scores(mc,Xte,CROWD))
    # payout regression on winning combos of training races
    Ftr=pay_feats(Xtr,Pc_tr); n=len(Xtr)//6
    rtr=np.where(trmask_races&ok)[0]
    cidx={tuple(cb):i for i,cb in enumerate(combos)}
    win=np.array([cidx[(sp[r,0]-1,sp[r,1]-1,sp[r,2]-1)] for r in rtr])
    rowsel=np.arange(n)*120+win
    yl=np.log(R_.payout.values[rtr]); Ft=Ftr.iloc[rowsel]
    PR=dict(objective='regression',learning_rate=0.05,num_leaves=15,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,verbose=-1,num_threads=4)
    mp=lgb.train(PR,lgb.Dataset(Ft,yl),250)
    resid=yl-mp.predict(Ft); smear=np.exp(resid).mean()   # in-sample smearing
    Fte=pay_feats(Xte,Pc_te); pp=np.exp(mp.predict(Fte)).reshape(-1,120)*smear
    rte=np.where(temask_races)[0]
    return rte,Pf_te,pp,Pc_te
cutR=(Rhd<cut)&(Rhd>='20250601'); teR=Rhd>=cut
if not WALK:
    rte,Pf,PP,Pc=run_period(cutR,teR)
    blocks=[(rte,Pf,PP,Pc)]
else:
    blocks=[]
    for m in sorted(set(Rmon[teR])):
        trm=(Rmon<m); tem=(Rmon==m); blocks.append(run_period(trm,tem)); print('walk',m,flush=True)
# ---- evaluation
cidx={tuple(cb):i for i,cb in enumerate(combos)}
rows=[]
for rte,Pf,PP,Pc in blocks:
    for k,r in enumerate(rte):
        pass
rte=np.concatenate([b[0] for b in blocks]); Pf=np.vstack([b[1] for b in blocks]); PP=np.vstack([b[2] for b in blocks]); Pc=np.vstack([b[3] for b in blocks])
win=np.array([cidx[(sp[r,0]-1,sp[r,1]-1,sp[r,2]-1)] if ok[r] else -1 for r in rte])
pay=np.where(ok[rte],R_.payout.values[rte],0.0)
okr=ok[rte]
EV=Pf*PP/100.0
print('テスト races',okr.sum(),'walk' if WALK else 'fixed')
# 全体の較正(120通り)
flat_p=Pf[okr].ravel(); hit=np.zeros_like(Pf[okr]); hit[np.arange(okr.sum()),win[okr]]=1; flat_h=hit.ravel()
q=pd.qcut(flat_p,10,duplicates='drop'); cal=pd.DataFrame({'p':flat_p,'h':flat_h}).groupby(q).agg(pred=('p','mean'),act=('h','mean'),n=('h','size')); print('較正 全120通り(decile)\n',cal.round(5))
# 払戻予測の精度: 当たった組の log 相関
wp=np.log(PP[okr][np.arange(okr.sum()),win[okr]]); wa=np.log(pay[okr]); print('当たり組のlog払戻 予測vs実績 相関',round(np.corrcoef(wp,wa)[0,1],3),' 予測平均',round(np.exp(wp).mean()),' 実績平均',round(np.exp(wa).mean()))
# 3連単の上位的中率
top1=Pf.argmax(1); print('モデル最有力組の的中率',round((top1[okr]==win[okr]).mean(),4))
mon=Rmon[rte]; hdte=Rhd[rte]
def evaluate(th,cap,tag):
    sel=[]
    for i in np.where(okr)[0]:
        ev=EV[i]; o=np.argsort(-ev)[:cap]; o=o[ev[o]>th]
        for j in o: sel.append((i,j))
    if not sel: print(tag,'bets 0'); return None
    s=pd.DataFrame(sel,columns=['i','j']); s['hit']=(win[s.i]==s.j); s['pay']=np.where(s.hit,pay[s.i],0); s['p']=Pf[s.i,s.j]; s['pp']=PP[s.i,s.j]; s['mon']=mon[s.i]; s['hd']=hdte[s.i]
    n=len(s); roi=s.pay.sum()/(100*n); ps=s.pay.sort_values(ascending=False)
    ex=lambda k:(s.pay.sum()-ps.head(k).sum())/(100*n)
    mm=s.groupby('mon').pay.agg(lambda x:x.sum()/(100*len(x))); 
    print(f'{tag} | bets {n} races {s.i.nunique()} hit {s.hit.sum()} ({s.hit.mean():.3f}) ROI {roi:.3f} 除top1 {ex(1):.3f} top3 {ex(3):.3f} top5 {ex(5):.3f} 勝ち月 {(mm>1).sum()}/{len(mm)} 最悪月 {mm.min():.2f} 平均予測的中 {s.p.mean():.4f} 平均予測払戻 {s.pp.mean():.0f} 実払戻/的中 {s.pay[s.hit].mean() if s.hit.any() else 0:.0f}')
    return s
res={}
for cap in (1,2):
    for th in (1.2,1.5,2.0):
        res[(cap,th)]=evaluate(th,cap,f'cap{cap} th{th}')
for th in (1.2,1.5,2.0):  # 払戻予測を使わない対照: 平均払戻=確定のみ... 群衆モデルの公正価格 0.75/pc を払戻とみなす
    pass
# 参照: 群衆価格 0.75/p_crowd で EV(=p_model/p_crowd*0.75)
EV2=Pf/np.maximum(Pc,1e-9)*0.75
def evaluate2(th,cap):
    sel=[]
    for i in np.where(okr)[0]:
        ev=EV2[i]; o=np.argsort(-ev)[:cap]; o=o[ev[o]>th]
        for j in o: sel.append((i,j))
    if not sel: print('crowdref none'); return
    s=pd.DataFrame(sel,columns=['i','j']); s['pay']=np.where(win[s.i]==s.j,pay[s.i],0); n=len(s); mm=s.groupby(mon[s.i]).pay.agg(lambda x:x.sum()/(100*len(x)))
    print(f'参照(群衆モデル価格, 払戻回帰なし) cap{cap} th{th} | bets {n} hit {(s.pay>0).sum()} ROI {s.pay.sum()/(100*n):.3f} 勝ち月 {(mm>1).sum()}/{len(mm)}')
for th in (1.2,1.5,2.0): evaluate2(th,1)
# 較正(bets)と月別・重なり
best=res[(2,1.5)]
for key_,s in res.items():
    if s is None or len(s)<50: continue
    q=pd.qcut(s.p,min(10,max(2,len(s)//30)),duplicates='drop'); cal=s.groupby(q).agg(pred=('p','mean'),act=('hit','mean'),n=('hit','size'))
    print('較正(bets)',key_); print(cal.round(4).T.to_string())
    if key_ in ((2,1.5),(1,1.5)):
        print(' 月別',s.groupby('mon').pay.agg(lambda x:round(x.sum()/(100*len(x)),2)).to_dict(),' 月別bets',s.groupby('mon').size().to_dict())
# 既存シグナルとの重なり
import io,contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(open('scripts/analysis/three_signals.py').read().split("S=pd.concat([S1,S2,S3])")[0].split("exec(open('scripts/analysis/one_pick.py')")[1].split('\n',1)[1])
S=pd.concat([S1,S2,S3])
S=S[S.hd>=cut]
rk=R_.reset_index().set_index(key)['index']
Sset=set()
for r in S.itertuples():
    t=tuple(int(x)-1 for x in r.cb.split('-')); Sset.add((rk.get((r.hd,r.jcd,r.rno)),cidx[t]))
for key_ in ((2,1.5),(1,1.5),(2,1.2),(2,2.0)):
    s=res[key_]
    if s is None: continue
    rr=rte[s.i.values]; ov=sum((int(a),int(b)) in Sset for a,b in zip(rr,s.j.values)); 
    rrace=set(S.merge(R_.iloc[rr][key].drop_duplicates(),on=key)[key].apply(tuple,axis=1)) 
    print('重なり',key_,'同一買い目',ov,'/',len(s),' 既存シグナル総数(テスト期間)',len(Sset))
