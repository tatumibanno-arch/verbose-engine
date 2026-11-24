"""自分基準（選手・モーター）との比較。一周などが「その選手/モーターのいつも」より良い/悪い日。オッズ不使用。仮説15本を先に固定。
基準=そのレースより厳密に前の開催日のみ。選手は直近10開催日(展示8回以上)、モーターは同場の直近20開催日(10回以上)。レース内z(小さい=速い)。"""
import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings('ignore')
src=open('scripts/analysis/three_signals.py').read().split("R=lambda s,col='pay'")[0]
exec(src)   # X, p, key, cut, S (3つの知らせ)
START='20250915'
ent=pd.read_sql("select hd,jcd,rno,lane,racer_no,motor_no from entries where hd>='20250601'",c)
X=X.merge(ent,on=key+['lane'],how='left')
for m in ('lap','mawari','chokusen'):
    g=X.groupby(key)[m]; X['z_'+m]=(X[m]-g.transform('mean'))/g.transform('std').replace(0,np.nan)
X['z_rl']=X.z_lap  # alias
def base(idn,metrics,K,minn,grp):
    D=X.dropna(subset=[idn]).copy(); out={}
    d=D.groupby(grp+['hd']).agg(**{**{f's_{m}':(m,'sum') for m in metrics},'n':('lap','count')}).reset_index().sort_values(grp+['hd'])
    gg=d.groupby(grp)
    for col in [f's_{m}' for m in metrics]+['n']:
        cs=gg[col].cumsum(); a=cs.groupby([d[k] for k in grp]).shift(1); b=cs.groupby([d[k] for k in grp]).shift(K+1)
        d['w_'+col]=a.fillna(0)-b.fillna(0)
    d.loc[d.w_n<minn,[f'w_s_{m}' for m in metrics]]=np.nan
    for m in metrics: d['b_'+idn[:2]+'_'+m]=d[f'w_s_{m}']/d.w_n
    return d[grp+['hd']+['b_'+idn[:2]+'_'+m for m in metrics]+['w_n']]
# z・rankの基準用に列を足す
for nm in ('lap','mawari','chokusen'): pass
X['zz_lap']=X.z_lap; X['zz_maw']=X.z_mawari; X['zz_cho']=X.z_chokusen; X['rr_lap']=X.r_lap
M=['zz_lap','zz_maw','zz_cho','rr_lap']
def base2(idn,K,minn,grp,tag):
    D=X.dropna(subset=[idn]).copy()
    d=D.groupby(grp+['hd']).agg(**{**{f's_{m}':(m,'sum') for m in M},'n':('lap','count')}).reset_index().sort_values(grp+['hd'])
    for col in [f's_{m}' for m in M]+['n']:
        cs=d.groupby(grp)[col].cumsum(); a=cs.groupby([d[k] for k in grp]).shift(1); b=cs.groupby([d[k] for k in grp]).shift(K+1)
        d['w_'+col]=a.fillna(0)-b.fillna(0)
    ok=d.w_n>=minn
    for m in M: d[f'{tag}_{m}']=np.where(ok,d[f'w_s_{m}']/d.w_n,np.nan)
    return d[grp+['hd']+[f'{tag}_{m}' for m in M]]
X=X.merge(base2('racer_no',10,8,['racer_no'],'rb'),on=['racer_no','hd'],how='left').merge(base2('motor_no',20,10,['jcd','motor_no'],'mb'),on=['jcd','motor_no','hd'],how='left')
for t in ('rb','mb'):
    for m in M: X[f'd{t}_{m}']=X[f'zz_{m[3:]}' if False else m]-X[f'{t}_{m}']
X=X[X.hd>=START].copy()
print('test boats',len(X),'with racer base',X.drb_zz_lap.notna().mean().round(3),'motor base',X.dmb_zz_lap.notna().mean().round(3))
# 艇ごとの順（公式展示順の並び）
Ex=X.pivot_table(index=key,columns='lane',values='et').reindex(columns=range(1,7))
ordr=pd.DataFrame(np.argsort(np.where(np.isnan(Ex.values),9,Ex.values),axis=1,kind='stable')+1,index=Ex.index)
O=ordr.reindex(pd.MultiIndex.from_frame(X[key])).values
s=X.lane.values[:,None]
def pick(excl_one):
    mask=(O==s)|((O==1)&excl_one) if excl_one else (O==s)
    res=[]; 
    keep=~mask; idx=np.argsort(~keep,axis=1,kind='stable')  # keep first
    Ok=np.take_along_axis(O,idx,axis=1); return Ok[:,0],Ok[:,1]
t_no1=pick(True); t_any=pick(False)
lane=X.lane.values
def combos(kind):
    L=lane.astype(str)
    if kind=='s-1-t': return [np.char.add(np.char.add(L,'-1-'),t_no1[0].astype(str))]
    if kind=='1-s-t': return [np.char.add(np.char.add('1-',L),'-'+t_no1[0].astype(str))]
    if kind=='1-a-b': a,b=t_any if False else pick_nos(); return [np.char.add(np.char.add('1-',a.astype(str)),np.char.add('-',b.astype(str)))]
    if kind=='fade': # 艇sが①のとき: ex最良の他艇h → h-1-t, h-t-1
        h,t=t_any
        return [np.char.add(np.char.add(h.astype(str),'-1-'),t.astype(str)) if False else np.char.add(h.astype(str),np.char.add('-1-',t.astype(str))),
                np.char.add(h.astype(str),np.char.add('-',np.char.add(t.astype(str),'-1')))]
def pick_nos():  # ①を除いた展示上位2(自艇sも含めて可): 1-a-b
    m=(O==1); idx=np.argsort(m,axis=1,kind='stable'); Ok=np.take_along_axis(O,idx,axis=1); return Ok[:,0],Ok[:,1]
a1,b1=pick_nos()
# 1-a-b は 展示順上位2艇(①以外)
def combo_1ab(): return [np.char.add('1-',np.char.add(a1.astype(str),np.char.add('-',b1.astype(str))))]
def combo_s1t(): return [np.char.add(lane.astype(str),np.char.add('-1-',t_no1[0].astype(str)))]
def combo_fade():
    h,t=t_any; hs,ts=h.astype(str),t.astype(str)
    return [np.char.add(hs,np.char.add('-1-',ts)),np.char.add(hs,np.char.add('-',np.char.add(ts,'-1')))]
CB={'s1t':combo_s1t,'1ab':combo_1ab,'fade':combo_fade}
pp=p.set_index(key+['res']).payout
def evaluate(nm,cond,kind,basecond):
    out=[]
    for tag,cnd in (('H',cond),('B',basecond)):
        m=np.asarray(cnd,bool); res=[]
        cs=CB[kind]()
        parts=[]
        for cc in cs:
            d=X.loc[m,key].copy(); d['cb']=cc[m]; parts.append(d)
        T=pd.concat(parts).drop_duplicates(key+['cb'])
        T=T.merge(p,left_on=key+['cb'],right_on=key+['res'],how='left',suffixes=('','_p')) if False else T.merge(p[key+['res','payout']].rename(columns={'res':'cb'}),on=key+['cb'],how='left')
        T['pay']=T.payout.fillna(0); out.append(T)
    return out
def rep(nm,T,Bt,tried):
    n=len(T); 
    if n<30: print(f'{nm} | n={n} 少ない'); return None
    pay=T.pay.values; r=lambda x,k: x.sum()/(100*k) if k else np.nan
    pj=T.sort_values('pay',ascending=False).pay.values; hf=T.hd.values<cut
    mo=T.groupby(T.hd.str[:6]).pay.apply(lambda s:s.sum()/(100*len(s)))
    ex=lambda k: (pay.sum()-pj[:k].sum())/(100*n)
    ov=T.merge(S[key+['cb']].assign(o=1),on=key+['cb'],how='left').o.notna().mean()
    ovr=T.merge(S[key].drop_duplicates().assign(o=1),on=key,how='left').o.notna().mean()
    print(f'{nm} | n={n} 的中{(pay>0).mean():.3f} 回収{pay.sum()/(100*n):.3f} 前半{r(pay[hf],hf.sum()):.3f}({hf.sum()}) 後半{r(pay[~hf],(~hf).sum()):.3f}({(~hf).sum()}) 除1/3/5 {ex(1):.3f}/{ex(3):.3f}/{ex(5):.3f} 勝ち月{(mo>1).sum()}/{len(mo)} | 基準(同買い方・条件なし)n={len(Bt)} 回収{Bt.pay.sum()/(100*len(Bt)):.3f} | 知らせと同買い目{ov:.2f} 同レース{ovr:.2f}')
A=X
lap_r=A.r_lap; nwlow=A.r_nw>=4; L26=A.lane>=2; L1=A.lane==1
Rz=A.drb_zz_lap; Mz=A.dmb_zz_lap
H=[
 ('H1 選手一周 自己比-1.0以下・勝率下位半分・②〜⑥・今日一周2位内 → s-1-t', (Rz<=-1)&nwlow&L26&(lap_r<=2),'s1t',nwlow&L26&(lap_r<=2)),
 ('H2 選手一周 自己比-1.5以下・②〜⑥ → s-1-t', (Rz<=-1.5)&L26,'s1t',L26),
 ('H3 ①の選手一周 自己比-1.0以下・今日一周2位内 → 1-a-b', (Rz<=-1)&L1&(lap_r<=2),'1ab',L1&(lap_r<=2)),
 ('H4 ①の選手が勝率1位なのに自己比+1.0以上・今日一周4位以下 → 他艇ex最良h: h-1-t,h-t-1', (Rz>=1)&L1&(A.r_nw<=1.5)&(lap_r>=4),'fade',L1&(A.r_nw<=1.5)&(lap_r>=4)),
 ('H5 モーター一周 自己比-1.0以下・②〜⑥・今日一周2位内 → s-1-t', (Mz<=-1)&L26&(lap_r<=2),'s1t',L26&(lap_r<=2)),
 ('H6 ①のモーター一周 自己比-1.0以下・今日一周2位内 → 1-a-b', (Mz<=-1)&L1&(lap_r<=2),'1ab',L1&(lap_r<=2)),
 ('H7 ①のモーター 自己比+1.0以上・今日一周4位以下 → h-1-t,h-t-1', (Mz>=1)&L1&(lap_r>=4),'fade',L1&(lap_r>=4)),
 ('H8 選手もモーターも自己比-0.7以下・②〜⑥ → s-1-t', (Rz<=-.7)&(Mz<=-.7)&L26,'s1t',L26),
 ('H9 選手の一周順位が基準より3つ以上良い・②〜⑥ → s-1-t', (A.drb_rr_lap<=-3)&L26,'s1t',L26),
 ('H10 選手まわり足 自己比-1.0以下・②〜⑥ → s-1-t', (A.drb_zz_maw<=-1)&L26,'s1t',L26),
 ('H11 選手直線 自己比-1.0以下・②〜⑥ → s-1-t', (A.drb_zz_cho<=-1)&L26,'s1t',L26),
 ('H12 選手の一周・まわり・直線の3つ中2つ以上 自己比-0.7以下・②〜⑥ → s-1-t', (((A.drb_zz_lap<=-.7).astype(int)+(A.drb_zz_maw<=-.7)+(A.drb_zz_cho<=-.7))>=2)&L26,'s1t',L26),
 ('H13 ①選手が勝率1位・自己比-0.8以下・一周2位内 → 1-a-b', (Rz<=-.8)&L1&(A.r_nw<=1.5)&(lap_r<=2),'1ab',L1&(A.r_nw<=1.5)&(lap_r<=2)),
 ('H14 ①が勝率4位以下・自己比-1.0以下・一周2位内 → 1-a-b', (Rz<=-1)&L1&(A.r_nw>=4)&(lap_r<=2),'1ab',L1&(A.r_nw>=4)&(lap_r<=2)),
 ('H15 選手の一周 自己比-1.0以下で「勝率最下位2艇」④〜⑥ → s-1-t', (Rz<=-1)&(A.lane>=4)&(A.r_nw>=5),'s1t',(A.lane>=4)&(A.r_nw>=5)),
]
print('試した数',len(H))
for nm,cond,kind,bc in H:
    T,Bt=evaluate(nm,cond.fillna(False)&(Rz.notna()|Mz.notna()),kind,bc)
    rep(nm,T,Bt,len(H))
if len(sys.argv)>1:
    T,Bt=evaluate('x',(Rz<=-1)&L26,'s1t',L26); print(T.head(), p.head(), T.dtypes, p.dtypes)
