"""フライング持ち（今の期にFをした選手）がオッズに入っているか。期=5/1〜10/31、11/1〜4/30。Fの記録は scripts/parse_result_codes.py の出力。
見方は extras_cal.py と同じ。2025-04〜2026-11、締切時オッズ。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
rc=pd.read_parquet(ob.D+'result_codes.parquet'); rc['d']=pd.to_datetime(rc.hd)
def term(d): return d.dt.year.astype(str)+np.where((d.dt.month>=5)&(d.dt.month<=10),'A',np.where(d.dt.month>=11,'B','')) 
rc['term']=np.where(rc.d.dt.month<=4,(rc.d.dt.year-1).astype(str)+'B',np.where(rc.d.dt.month<=10,rc.d.dt.year.astype(str)+'A',rc.d.dt.year.astype(str)+'B'))
f=rc[rc.code=='F'][['racer_no','hd','term','d']].sort_values('d')
e=pd.read_sql("select hd,jcd,rno,lane,racer_no from entries where hd>='20250401'",c); e['racer_no']=e.racer_no.astype(int); e['d']=pd.to_datetime(e.hd)
e['term']=np.where(e.d.dt.month<=4,(e.d.dt.year-1).astype(str)+'B',np.where(e.d.dt.month<=10,e.d.dt.year.astype(str)+'A',e.d.dt.year.astype(str)+'B'))
# その日より前の、同じ期のF本数と、最後のFからの日数
fd=f.groupby(['racer_no','term','hd']).size().rename('n').reset_index().sort_values('hd'); 
m=e.merge(fd,on=['racer_no','term'],how='left',suffixes=('','_f')); m=m[(m.hd_f.isna())|(m.hd_f<m.hd)]
agg=m.groupby(key+['lane']).agg(fn=('n','sum'),lastf=('hd_f','max')).reset_index(); e=e.merge(agg,on=key+['lane'],how='left'); e['fn']=e.fn.fillna(0)
e['since']=(e.d-pd.to_datetime(e.lastf)).dt.days
st=rc[['hd','jcd','rno','lane','st']]; e=e.merge(st,on=key+['lane'],how='left')
print('F持ちの割合',(e.fn>=1).mean().round(3),'F2以上',(e.fn>=2).mean().round(4))
print('本番の平均スタート: F無し',e[e.fn==0].st.clip(lower=0).mean().round(3),'F1',e[e.fn==1].st.clip(lower=0).mean().round(3),'F2以上',e[e.fn>=2].st.clip(lower=0).mean().round(3),' F後30日以内',e[(e.since<=30)].st.clip(lower=0).mean().round(3))
R,L,oc=ob.load(); A=R[key+['res','tot']+oc].copy(); A['p1']=(1/A[oc]).sum(1)/A.tot; A['win1']=(A.res.str[0]=='1').astype(float)
F=e[key+['lane','fn','since']]; L=L.merge(F,on=key+['lane'],how='left'); W=L[L.win1&(L.p1>=.4)]; A=A.merge(F[F.lane==1],on=key,how='left')
def show(nm,s,b,mk,ac,base):
    s=s.assign(b=b).dropna(subset=['b']); s=s[s.b!='']; t=s.groupby('b',observed=True).agg(n=(ac,'size'),mk=(mk,'mean'),ac=(ac,'mean')); t['差']=t.ac-t.mk
    for nm2,q in (('前半',s[s.hd<'20260401']),('後半',s[s.hd>='20260401'])): u=q.groupby('b',observed=True).agg(mk=(mk,'mean'),ac=(ac,'mean')); t[nm2]=u.ac-u.mk
    t.columns=['件数','売れ方','実際','差','前半の差','後半の差']; print(f'\n{nm}（全体の差 {base:+.3f}）'); print(t.round(3).to_string())
lab=lambda x: np.where(x.fn>=2,'F2以上',np.where(x.fn==1,'F1','F無し'))
show('[相手] F持ち',W,lab(W),'mk3','in3',W.in3.mean()-W.mk3.mean()); show('[1号艇] F持ち',A,lab(A),'p1','win1',A.win1.mean()-A.p1.mean())
sl=lambda x: np.where(x.fn==0,'',np.where(x.since<=14,'Fから14日以内',np.where(x.since<=45,'15〜45日','46日以上')))
show('[相手] 最後のFからの日数',W,sl(W),'mk3','in3',W.in3.mean()-W.mk3.mean()); show('[1号艇] 最後のFからの日数',A,sl(A),'p1','win1',A.win1.mean()-A.p1.mean())
# 相手にF持ちが何人いるかで1号艇
cnt=e[e.lane>=2].assign(h=(e.fn>=1)).groupby(key).h.sum().rename('nf').reset_index(); A2=A.merge(cnt,on=key,how='left')
show('[1号艇] 相手②〜⑥のF持ちの人数',A2,A2.nf.clip(upper=3).map({0:'0人',1:'1人',2:'2人',3:'3人以上'}),'p1','win1',A.win1.mean()-A.p1.mean())
c2=e[e.lane==2].assign(f2=(e.fn>=1))[key+['f2']]; A3=A.merge(c2,on=key,how='left'); show('[1号艇] 2号艇がF持ち',A3,np.where(A3.f2==True,'2号艇がF持ち','2号艇はF無し'),'p1','win1',A.win1.mean()-A.p1.mean())
