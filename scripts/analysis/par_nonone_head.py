"""①がオリ展で弱く、外（2-4）の1艇が明確に最良のレースで、その艇を頭に買う。1-3点/6点。払戻は確定（オッズ不使用）。"""
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
import itertools
rng=np.random.default_rng(0)
Xp=X.copy(); Xp['sc']=Xp.r_lap+Xp.r_maw+Xp.r_cho+Xp.r_ex
def mat(col):
    m=Xp.pivot_table(index=key,columns='lane',values=col,aggfunc='first'); return m
L=mat('lap'); S=mat('sc'); RL=mat('r_lap')
idx=L.index.intersection(S.index); L=L.loc[idx]; S=S.loc[idx]; RL=RL.loc[idx]
ok=(L.notna().all(1)&S.notna().all(1)&(L.columns.size==6)).values
L=L.values[ok]; S=S.values[ok]; RL=RL.values[ok]; idx=idx[ok]
pp=p.set_index(key); pm=pp.reindex(idx); res=pm.res.values; pay=pm.payout.values
hd=np.array([i[0] for i in idx]); jcd=np.array([i[1] for i in idx])
N=len(idx); print('races',N,'lap gap quantiles',np.nanquantile(np.sort(L,1)[:,1]-np.sort(L,1)[:,0],[.25,.5,.75,.9]),'score gap q',np.quantile(np.sort(S,1)[:,1]-np.sort(S,1)[:,0],[.25,.5,.75,.9]))
Sk=S+L*1e-3  # tie-break by lap
def orders(K): return np.argsort(K,1,kind='stable')  # lane index 0..5
OL=orders(L); OS=orders(Sk)
rk=lambda K: np.argsort(np.argsort(K,1,kind='stable'),1)
RKl=rk(L); RKs=rk(Sk)
def tickets(h,o,kind):
    others=[x for x in o if x!=h]; non1=[x for x in others if x!=0]
    f=lambda a:'-'.join(str(x+1) for x in a)
    if kind=='T1': a=others[:2]; return [f((h,a[0],a[1]))]
    if kind=='T2': a=others[:2]; return [f((h,a[0],a[1])),f((h,a[1],a[0]))]
    if kind=='T3': a=others[:3]; return [f((h,x,y)) for x in a for y in a if x!=y]
    if kind=='T4': a=non1[:3]; return [f((h,x,y)) for x in a for y in a if x!=y]
    if kind=='T5': a=non1[:2]; return [f((h,a[0],a[1]))]
    if kind=='T6': a=non1[:2]; return [f((h,a[0],a[1])),f((h,a[1],a[0]))]
def stats(sel,tk,Ob,nm):
    bets=0; pays=[]; rows=[]; combos={}
    for i in sel:
        h=Hh[i]; ts=tickets(h,Ob[i],tk); combos[i]=ts; bets+=len(ts)
        if res[i] in ts: pays.append(pay[i]); rows.append(i)
    return bets,combos
def roi(sel,combos,bets,mask=None):
    pass
def evalv(sel,combos,pr=pay,rs=res):
    cost=0; tot=0; per=[]
    for i in sel:
        c=combos[i]; cost+=100*len(c); w=pr[i] if rs[i] in c else 0; tot+=w; per.append(w)
    return cost,tot,per
variants=[]; out=[]
weak={'W0none':lambda: np.ones(N,bool),'W1①周≥4位':lambda: RKl[:,0]>=3,'W2①総合≥4位':lambda: RKs[:,0]>=3}
strong=[('lap',g) for g in (0.05,0.1,0.2)]+[('sc',g) for g in (2,4,6)]
for wn,wf in weak.items():
    w=wf()
    for (sk,g) in strong:
        K=L if sk=='lap' else Sk; O=OL if sk=='lap' else OS
        srt=np.sort(K,1); gap=srt[:,1]-srt[:,0]; Hh=O[:,0]
        base=w&(Hh>=1)&(Hh<=3)&(gap>=g)&(~np.isnan(gap))
        sel=np.where(base)[0]
        for tk in ('T1','T2','T3','T4','T5','T6'):
            combos={i:tickets(Hh[i],O[i],tk) for i in sel}
            cost,tot,per=evalv(sel,combos)
            if cost==0: continue
            per=np.array(per); ii=np.array(sel)
            m=hd[ii]<str(cut); 
            c1=sum(100*len(combos[i]) for i in ii[m]); c2=cost-c1
            r1=per[m].sum()/c1 if c1 else np.nan; r2=per[~m].sum()/c2 if c2 else np.nan
            sp=np.sort(per)[::-1]; ex=[(tot-sp[:k].sum())/cost for k in (1,3,5)]
            mon=pd.DataFrame({'m':[hd[i][:6] for i in ii],'c':[100*len(combos[i]) for i in ii],'p':per}).groupby('m').sum()
            wm=(mon.p>mon.c).sum()
            out.append(dict(nm=f'{wn}|{sk}{g}|{tk}',races=len(sel),bets=cost//100,roi=tot/cost,r1=r1,r2=r2,ex1=ex[0],ex3=ex[1],ex5=ex[2],wm=f'{wm}/{len(mon)}',hit=(per>0).mean(),sel=sel,combos=combos))
print('variants tried',len(out))
df=pd.DataFrame(out).drop(columns=['sel','combos']).sort_values('roi',ascending=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(df.head(20).round(3).to_string())
print('median roi',df.roi.median().round(3),'n roi>1',(df.roi>1).sum(),'both halves>1',((df.r1>1)&(df.r2>1)).sum())
prom=df[(df.r1>1)&(df.r2>1)&(df.ex3>1)&(df.bets>=150)]; print('promising pre-fake:\n',prom.round(3).to_string())
# fake world for best (by roi among bets>=150), plus best promising if any
cand=df[df.bets>=150]
if len(cand):
    nm=(prom if len(prom) else cand).iloc[0].nm; best=[o for o in out if o['nm']==nm][0]
    sel=best['sel']; combos=best['combos']; real=best['roi']
    ym=np.array([j+h[:6] for j,h in zip(jcd,hd)]); grp=pd.Series(np.arange(N)).groupby(ym).apply(lambda s:s.values).values
    cnt=0; fr=[]
    for _ in range(200):
        perm=np.arange(N)
        for g_ in grp: perm[g_]=rng.permutation(g_)
        pr=pay[perm]; rs=res[perm]; cost,tot,_=evalv(sel,combos,pr,rs); fr.append(tot/cost); cnt+= (tot/cost>=real)
    print('FAKE best',nm,'real',round(real,3),'fake mean',round(np.mean(fr),3),'sd',round(np.std(fr),3),'share>=real',cnt/200)
