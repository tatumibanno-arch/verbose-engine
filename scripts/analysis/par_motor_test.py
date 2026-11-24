"""オレンジモーター仮説検証。オッズ不使用。15仮説。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/pm/'
F=pd.read_parquet(D+'feat.parquet'); F=F[F.hd>='20250601']
X=X.merge(F[key+['lane','pos','m10_t2r','m10_res','m20_res','meet_res','meet_t2r','meet_n','m10_n']],on=key+['lane'],how='left')
# 予測力チェック: 調子の残差(m10_res)の五分位ごとの当日平均着順（枠・勝率補正なしの素の差）
Z=X.dropna(subset=['m10_res','pos']); Z['q']=pd.qcut(Z.m10_res,5,labels=False)
print('m10_res五分位 -> 平均着順/1着率/2連対率'); print(Z.groupby('q').pos.agg(['mean',lambda s:(s==1).mean(),lambda s:(s<=2).mean()]).round(3).to_string())
Z=X.dropna(subset=['mo','pos']); Z['q']=pd.qcut(Z.mo,5,labels=False,duplicates='drop'); print('season mo五分位'); print(Z.groupby('q').pos.agg(['mean',lambda s:(s<=2).mean()]).round(3).to_string())
X['f10']=X.m10_res.where(X.m10_n>=8); X['fm']=X.meet_res.where(X.meet_n>=6)
for q in ('f10','fm'): X['r_'+q]=X.groupby(key)[q].rank(method='average')   # 小さい=好調(着順残差が負)
hot10=X.f10.quantile(.2); cold10=X.f10.quantile(.8); hotm=X.fm.quantile(.2); coldm=X.fm.quantile(.8)
print('閾値 hot10',round(hot10,3),'cold10',round(cold10,3),'hotm',round(hotm,3),'coldm',round(coldm,3))
W=lambda col: X.pivot_table(index=key,columns='lane',values=col)
lap=W('r_lap'); f10=W('f10').reindex(index=lap.index,columns=range(1,7)); fm=W('fm').reindex(index=lap.index,columns=range(1,7)); mo=W('r_mo').reindex(index=lap.index,columns=range(1,7)); lap=lap.reindex(columns=range(1,7))
idx=lap.index; M=pd.DataFrame(index=idx)
lapv=lap.values; 
# 非①艇を一周順位で並べた最良・次点
oth=lapv.copy(); oth[:,0]=np.inf
o=np.argsort(oth,axis=1,kind='stable')  # lane index 0-based
A=o[:,0]+1; B=o[:,1]+1; C=o[:,2]+1
f10v=f10.values; fmv=fm.values; nrow=len(idx); ar=np.arange(nrow)
l1=lapv[:,0]
# 非①で調子最良(f10最小)の艇
f_oth=f10v.copy(); f_oth[:,0]=np.inf; f_oth=np.where(np.isnan(f_oth),np.inf,f_oth); H=np.argmin(f_oth,axis=1); Hv=f_oth[ar,H]; H=H+1
fm_oth=fmv.copy(); fm_oth[:,0]=np.inf; fm_oth=np.where(np.isnan(fm_oth),np.inf,fm_oth); HM=np.argmin(fm_oth,axis=1); HMv=fm_oth[ar,HM]; HM=HM+1
lapH=lapv[ar,H-1]; lapHM=lapv[ar,HM-1]
def other(x):  # xとは別の「一周良い非①」
    return np.where(A==x,B,A)
s=lambda a: a.astype(str)
cb1AB='1-'+s(A)+'-'+s(B)
res=p.set_index(key).reindex(idx)
R=pd.DataFrame({'res':res.res.values,'payout':res.payout.values,'hd':idx.get_level_values('hd')},index=idx)
hot1=f10v[:,0]<=hot10; cold1=f10v[:,0]>=cold10; hotm1=fmv[:,0]<=hotm; coldm1=fmv[:,0]>=coldm
mo1=mo.values[:,0]
tests=[]
def T(nm,mask,cb): tests.append((nm,np.asarray(mask,bool)&pd.notna(R.res.values),np.asarray(cb,object)))
T('H0 対照: ①一周1位 → 1-A-B',l1==1,cb1AB)
T('H1 ①好調(10走)&一周≤2 → 1-A-B',hot1&(l1<=2),cb1AB)
T('H2 ①好調&一周1位 → 1-A-B',hot1&(l1==1),cb1AB)
T('H3 ①好調 → 1-A-B(展示無視)',hot1,cb1AB)
T('H4 ①不調&一周≥4 → A-B-C(①外し)',cold1&(l1>=4),s(A)+'-'+s(B)+'-'+s(C))
T('H5 ①不調&一周≥4 → A-1-B',cold1&(l1>=4),s(A)+'-1-'+s(B))
T('H6 ①不調(10走) → A-B-C',cold1,s(A)+'-'+s(B)+'-'+s(C))
hotH=(Hv<=hot10)
T('H7 非①好調艇Hが一周1位 → 1-H-別',hotH&(lapH==1)&(H==A),'1-'+s(H)+'-'+s(B))
T('H8 非①好調艇Hが一周≤2 → 1-H-別',hotH&(lapH<=2)&(l1<=3),'1-'+s(H)+'-'+s(other(H)))
T('H9 非①好調H一周1位&①不調 → H-1-別',hotH&(lapH==1)&(H==A)&cold1,s(H)+'-1-'+s(B))
T('H10 非①好調H一周1位(①問わず) → H-1-別',hotH&(lapH==1)&(H==A),s(H)+'-1-'+s(B))
T('H11 節内: ①節内好調&一周≤2 → 1-A-B',(fmv[:,0]<=hotm)&(l1<=2),cb1AB)
T('H12 節内: 非①節内好調HM一周1位 → 1-HM-別',(HMv<=hotm)&(lapHM==1)&(HM==A),'1-'+s(HM)+'-'+s(B))
T('H13 節内: ①節内不調&一周≥4 → A-1-B',coldm1&(l1>=4),s(A)+'-1-'+s(B))
T('H14 ①好調だがシーズン2連率は下位(順位≥4)&一周≤2 → 1-A-B',hot1&(mo1>=4)&(l1<=2),cb1AB)
T('H15 ①シーズン2連率1位だが直近不調&一周≤2 → 1-A-B(逆指標)',cold1&(mo1==1)&(l1<=2),cb1AB)
print('試した仮説数',len(tests))
print(f'{"仮説":<52}{"n":>6}{"的中":>6}{"回収":>7}{"前半":>7}{"後半":>7}{"除top1":>7}{"除top3":>7}{"勝月":>7}')
for nm,m,cb in tests:
    r=R[m].copy(); r['pay']=np.where(r.res.values==cb[m],r.payout.values,0)
    n=len(r)
    if n<80: print(nm,n,'少ない'); continue
    a=r.hd.values<cut; roi=lambda x:x.sum()/(100*max(len(x),1))
    pv=np.sort(r.pay.values)[::-1]
    mo_=r.groupby(r.hd.str[:6]).pay.apply(lambda x:x.sum()/(100*len(x)))
    print(f'{nm:<52}{n:>6}{(r.pay>0).mean():>6.3f}{roi(r.pay.values):>7.3f}{roi(r.pay.values[a]):>7.3f}{roi(r.pay.values[~a]):>7.3f}{(pv[1:].sum()/(100*n)):>7.3f}{(pv[3:].sum()/(100*n)):>7.3f}{(mo_>1).sum():>4}/{len(mo_)}')
