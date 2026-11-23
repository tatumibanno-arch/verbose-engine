"""オリジナル展示（一周・まわり足・直線、BOATCAST）がオッズに入っているか。公式の展示タイムの順位で揃えたうえでも差が残るかも見る。
数字はレース内の順位（タイムが小さいほど上位）。締切時オッズ。期間はオリジナル展示が取れた範囲（2025-06〜）。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
o=pd.read_sql("select hd,jcd,rno,lane,lap,mawari,chokusen,status from oriten",c)
for q in ('lap','mawari','chokusen'): o[q+'_rk']=o.groupby(key)[q].rank(method='min')
E=pd.read_parquet(ob.D+'ob_ent.parquet')[key+['lane','er']]; o=o.merge(E,on=key+['lane'],how='left')
R,L,oc=ob.load(); A=R[key+['res','tot']+oc].copy(); A['p1']=(1/A[oc]).sum(1)/A.tot; A['win1']=(A.res.str[0]=='1').astype(float)
A=A.merge(o[o.lane==1],on=key); L=L.merge(o.drop(columns=['er']),on=key+['lane']); W=L[L.win1&(L.p1>=.4)]
cut=sorted(A.hd.tolist())[len(A)//2]; print('レース',len(A),'期間',A.hd.min(),A.hd.max(),'前半/後半の境',cut)
def show(nm,s,b,mk,ac):
    s=s.assign(b=b).dropna(subset=['b']); t=s.groupby('b',observed=True).agg(n=(ac,'size'),m=(mk,'mean'),a=(ac,'mean')); t['差']=t.a-t.m
    for nm2,q in (('前半',s[s.hd<cut]),('後半',s[s.hd>=cut])): u=q.groupby('b',observed=True).agg(m=(mk,'mean'),a=(ac,'mean')); t[nm2]=u.a-u.m
    t.columns=['件数','売れ方','実際','差','前半の差','後半の差']; print('\n'+nm+f'（全体の差 {s[ac].mean()-s[mk].mean():+.3f}）'); print(t.round(3).to_string())
for q,nm in (('lap','一周（半周）'),('mawari','まわり足'),('chokusen','直線')):
    show(f'[1号艇] {nm}の順位',A,A[q+'_rk'].clip(upper=5).map({1:'1位',2:'2位',3:'3位',4:'4位',5:'5〜6位'}),'p1','win1')
    show(f'[相手] {nm}の順位',W,W[q+'_rk'].clip(upper=5).map({1:'1位',2:'2位',3:'3位',4:'4位',5:'5〜6位'}),'mk3','in3')
print('\n===== 公式展示の順位で揃えたうえで、オリジナル展示がどう効くか')
for q,nm in (('lap','一周'),('mawari','まわり足')):
    lab=lambda d: np.where(d.er<=2,'公式1〜2位',np.where(d.er>=5,'公式5〜6位','公式3〜4位')).astype(object)+'×'+np.where(d[q+'_rk']<=2,nm+'1〜2位',np.where(d[q+'_rk']>=5,nm+'5〜6位',nm+'3〜4位')).astype(object)
    show(f'[1号艇] 公式展示×{nm}',A,lab(A),'p1','win1'); show(f'[相手] 公式展示×{nm}',W,lab(W),'mk3','in3')
