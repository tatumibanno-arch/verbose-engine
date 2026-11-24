"""条件の組み合わせ（2つ・3つ）×固定の買い目8種を、発見期間(hd<20260201)だけで探し、上位30を検証期間に当てる。
同じ手順を「レースの結果を同じ場・同じ月の中でシャッフルした偽の世界」でも繰り返し、本物と比べる。オッズは使わない。"""
import sqlite3, numpy as np, pandas as pd, itertools, time, sys, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cutn=int(cut); t0=time.time()
NW=int(sys.argv[1]) if len(sys.argv)>1 else 20
# ---- race table
def piv(col):
    t=X.pivot_table(index=key,columns='lane',values=col,aggfunc='first'); return t
base=piv('r_lap'); idx=base.index
G={c:piv(c).reindex(idx).reindex(columns=range(1,7)).values for c in ['r_lap','r_maw','r_cho','r_ex','r_st','r_nw','r_mo','r_c2','et','lap']}
rc=idx.to_frame(index=False)
ent=pd.read_sql("select hd,jcd,rno,race_name,close_time,grade from entries where hd>='20250601' and lane=1",c).drop_duplicates(key)
bi=pd.read_sql("select hd,jcd,rno,json_extract(data,'$.wind_speed') wind,json_extract(data,'$.wave_height') wave from beforeinfo where hd>='20250601' and lane=1",c).drop_duplicates(key)
rc=rc.merge(ent,on=key,how='left').merge(bi,on=key,how='left').merge(p,on=key,how='left')
ok=rc.res.notna().values
rc=rc[ok].reset_index(drop=True); G={k:v[ok] for k,v in G.items()}
N=len(rc); hd=rc.hd.values; D=(hd.astype(int)<cutn)
print('races',N,'disc',D.sum(),'valid',(~D).sum(),flush=True)
L=lambda k,j:G[k][:,j-1]
rn=rc.race_name.fillna(''); ct=rc.close_time.fillna('00:00'); rno=rc.rno.astype(int).values
cond={}
lap_sorted=np.sort(G['lap'],axis=1)
cond['lap1_r1']=L('r_lap',1)==1; cond['lap1_le2']=L('r_lap',1)<=2; cond['lap1_ge4']=L('r_lap',1)>=4
cond['maw1_r1']=L('r_maw',1)==1; cond['cho1_le2']=L('r_cho',1)<=2
cond['top3lap_123']=np.nan_to_num(G['r_lap'][:,:3],nan=9).max(1)<=3
cond['lapgap34_ge.3']=(lap_sorted[:,3]-lap_sorted[:,2])>=0.3
cond['lap2_r1']=L('r_lap',2)==1; cond['lap2_le2']=L('r_lap',2)<=2; cond['lap3_le3']=L('r_lap',3)<=3; cond['lap23_le3']=(L('r_lap',2)<=3)&(L('r_lap',3)<=3)
cond['maw2_le2']=L('r_maw',2)<=2; cond['cho2_le2']=L('r_cho',2)<=2; cond['cho3_le2']=L('r_cho',3)<=2
cond['ex1_r1']=L('r_ex',1)==1; cond['ex1_le2']=L('r_ex',1)<=2; cond['ex1_ge5']=L('r_ex',1)>=5; cond['ex2_le2']=L('r_ex',2)<=2; cond['ex23_le3']=(L('r_ex',2)<=3)&(L('r_ex',3)<=3)
cond['st1_le2']=L('r_st',1)<=2; cond['st1_ge5']=L('r_st',1)>=5
cond['nw1_r1']=L('r_nw',1)==1; cond['nw1_ge3']=L('r_nw',1)>=3; cond['mo1_r1']=L('r_mo',1)==1; cond['mo1_le2']=L('r_mo',1)<=2; cond['mo1_ge3']=L('r_mo',1)>=3; cond['c2_1_r1']=L('r_c2',1)==1
cond['wind>=4']=(pd.to_numeric(rc.wind,errors='coerce')>=4).values; cond['wave>=5']=(pd.to_numeric(rc.wave,errors='coerce')>=5).values
cond['cho6_r1']=L('r_cho',6)==1
cond['R1-4']=rno<=4; cond['R5-8']=(rno>=5)&(rno<=8); cond['R9-12']=rno>=9
cond['night(close>=17:30)']=(ct>='17:30').values
cond['予選']=rn.str.contains('予選').values; cond['準優']=rn.str.contains('準優').values; cond['優勝戦']=(rn.str.contains('優勝')&~rn.str.contains('準優')).values; cond['選抜']=rn.str.contains('選抜').values
cond['1A1']=(rc.grade=='A1').values
names=list(cond); M=np.column_stack([np.asarray(cond[k],dtype=bool) for k in names]); C=len(names)
print(C,'conditions; base rates',dict(zip(names,M.mean(0).round(2))),flush=True)
# ---- tickets
et=np.nan_to_num(G['et'],nan=9.0); lap=np.where(np.isnan(G['lap']),99,G['lap'])
lane_idx=np.arange(1,7)
l5=lap[:,1:]; o=np.argsort(l5,axis=1,kind='stable')[:,:2]; A=o[:,0]+2; B=o[:,1]+2
eA=et[np.arange(N),A-1]; eB=et[np.arange(N),B-1]; sw=eA>eB; A2=np.where(sw,B,A); B2=np.where(sw,A,B)
f23=et[:,1]<=et[:,2]
code=lambda a,b,c_:a*100+b*10+c_
TK={'1-2-3':code(1,2,3)+0*rno,'1-3-2':code(1,3,2)+0*rno,'1-exfast-other':np.where(f23,123,132),'1-A-B(lap top2 by ex)':code(1,A2,B2),'2-1-3':213+0*rno,'3-1-2':312+0*rno,'1-2-4':124+0*rno,'1-3-4':134+0*rno}
tn=list(TK); T=len(tn); CB=np.column_stack([TK[k] for k in tn])
resc=np.array([int(s.replace('-','')) for s in rc.res]); payv=rc.payout.values.astype(float)
# ---- combos
combos=[(i,j) for i,j in itertools.combinations(range(C),2)]+list(itertools.combinations(range(C),3))
combos=np.array([list(x)+[-1]*(3-len(x)) for x in combos])
def cmask(cb,rows):
    m=M[rows][:,cb[0]]&M[rows][:,cb[1]]
    if cb[2]>=0: m&=M[rows][:,cb[2]]
    return m
Di=np.where(D)[0]; Vi=np.where(~D)[0]
# discovery masks (combos x Nd) float32, dedupe identical race sets
Md=np.empty((len(combos),len(Di)),dtype=np.float32)
Md_b=M[Di]
for k,cb in enumerate(combos):
    m=Md_b[:,cb[0]]&Md_b[:,cb[1]]
    if cb[2]>=0: m&=Md_b[:,cb[2]]
    Md[k]=m
nd=Md.sum(1); keep=nd>=150
seen={}; 
for k in np.where(keep)[0]:
    h=hash(Md[k].astype(bool).tobytes())
    if h in seen: keep[k]=False
    else: seen[h]=k
print('combos',len(combos),'n>=150 & unique masks',keep.sum(),flush=True)
KI=np.where(keep)[0]; MdK=Md[KI]; ndK=nd[KI]; del Md
def mname(cb): return ' & '.join(names[x] for x in cb if x>=0)
def search(resc_,pay_,K=2500):
    PM=np.where(resc_[:,None]==CB,pay_[:,None],0.0)   # N x T
    Pd=PM[Di]; S=MdK.astype(np.float64)@Pd if False else (MdK@Pd.astype(np.float32)).astype(np.float64)
    roi=S/(100*ndK[:,None]); flat=np.argsort(-roi,axis=None)[:K]
    rows=[]
    for f in flat:
        a,t=divmod(f,T); sel=np.nonzero(MdK[a])[0]; pv=Pd[sel,t]; srt=np.sort(pv)[::-1]
        ex3=(pv.sum()-srt[:3].sum())/(100*len(sel)); rows.append((ex3,roi[a,t],a,t))
    bound=roi.flat[flat[-1]]
    rows.sort(key=lambda r:-r[0]); top=rows[:30]
    return top,bound,PM
def validate(top,PM):
    out=[]
    for ex3,r,a,t in top:
        cb=combos[KI[a]]; m=cmask(cb,Vi); nv=m.sum(); pv=PM[Vi][m,t]
        out.append((nv,pv.sum()/(100*nv) if nv else np.nan,(pv>0).mean() if nv else np.nan))
    return out
t1=time.time()
top,bound,PM=search(resc,payv)
print('real search',round(time.time()-t1,1),'s; 30th ex3',round(top[-1][0],3),'vs raw bound of K-th',round(bound,3),'(ok if 30th ex3 >= bound)',flush=True)
val=validate(top,PM)
mon=rc.hd.str[:6].values
def fullstats(cb,t,PM_):
    m=cmask(cb,np.arange(N)); pv=PM_[m,t]; n=m.sum(); srt=np.sort(pv)[::-1]; R_=lambda s:s/(100*n)
    mo=pd.Series(pv).groupby(mon[m]).apply(lambda s:s.sum()/(100*len(s)))
    return n,(pv>0).mean(),R_(pv.sum()),R_(pv.sum()-srt[:1].sum()),R_(pv.sum()-srt[:3].sum()),R_(pv.sum()-srt[:5].sum()),(mo>1).sum(),len(mo),m
# signals overlap
src=open('scripts/analysis/three_signals.py').read(); exec(src.split('S=pd.concat')[0].split("exec(open('scripts/analysis/one_pick.py')")[1].split('\n',1)[1])
SG=pd.concat([S1,S2,S3])[key+['cb']].drop_duplicates(); SG['cbc']=SG.cb.str.replace('-','').astype(int)
sigset=set(zip(SG.hd,SG.jcd,SG.rno,SG.cbc)); sigrace=set(zip(SG.hd,SG.jcd,SG.rno))
print('\n=== REAL: top30 by discovery ex-top3 ROI')
print('rank | rule | ticket | disc n ex3 raw | valid n hit ROI | full n hit ROI ex1 ex3 ex5 winmo | overlap w/ live signals (same ticket / same race)')
nval=0; rk=0
for (ex3,r,a,t),(nv,vr,vh) in zip(top,val):
    rk+=1; cb=combos[KI[a]]; n,h,R0,R1,R3,R5,w,tm,m=fullstats(cb,t,PM)
    ii=np.nonzero(m)[0]; same=sum((hd[i],rc.jcd[i],rno[i],CB[i,t]) in sigset for i in ii) if False else sum((rc.hd[i],rc.jcd[i],rc.rno[i],int(CB[i,t])) in sigset for i in ii); rr=sum((rc.hd[i],rc.jcd[i],rc.rno[i]) in sigrace for i in ii)
    nval+=vr>1
    print(f'{rk:2d} | {mname(cb)} | {tn[t]} | {int(ndK[a])} {ex3:.3f} {r:.3f} | {nv} {vh:.3f} {vr:.3f} | {n} {h:.3f} {R0:.3f} {R1:.3f} {R3:.3f} {R5:.3f} {w}/{tm} | {same}/{n} {rr}/{n}')
print('REAL validate ROI>1:',nval,'/30; mean valid ROI',np.nanmean([v[1] for v in val]).round(3),'; mean disc ex3',np.mean([x[0] for x in top]).round(3),flush=True)
# ---- fake worlds
grp=(rc.jcd+'_'+rc.hd.str[:6]).values; gi=pd.Series(np.arange(N)).groupby(grp).apply(np.array).values
rng=np.random.default_rng(12345); res_rows=[]
for wld in range(NW):
    perm=np.arange(N)
    for g_ in gi: perm[g_]=rng.permutation(g_)
    tp,bd,PMf=search(resc[perm],payv[perm]); vf=validate(tp,PMf)
    nv_=sum(v[1]>1 for v in vf); res_rows.append((nv_,np.nanmean([v[1] for v in vf]),np.mean([x[0] for x in tp]),np.mean([v[0] for v in vf]),max(v[1] for v in vf if v[1]==v[1])))
    print(f'fake {wld}: validate>1 {nv_}/30 meanValROI {res_rows[-1][1]:.3f} meanDiscEx3 {res_rows[-1][2]:.3f} meanValN {res_rows[-1][3]:.0f} maxValROI {res_rows[-1][4]:.2f} ({time.time()-t0:.0f}s)',flush=True)
F=np.array(res_rows)
print(f'\nFAKE avg validate>1: {F[:,0].mean():.1f}/30 (sd {F[:,0].std():.1f}, min {F[:,0].min():.0f}, max {F[:,0].max():.0f}); mean valROI {F[:,1].mean():.3f}; mean discEx3 {F[:,2].mean():.3f} (REAL {np.mean([x[0] for x in top]):.3f}); REAL validate>1 {nval}')
print('total sec',round(time.time()-t0))
