"""まだ見ていなかった材料が、オッズに入っているかを一気に見る（チルト・体重・1日2走目・風向き・気温・勝負駆け）。
見方: ①頭の売れ方と実際の1着率の差、①が勝ったレースで相手が3着以内に入る売れ方と実際の差。差が全体の平均からずれていれば入り残し。
2025-04〜2026-11、締切時オッズ（締切前とは違う）。前半=〜2026-03、後半=2026-04〜。"""
import sqlite3, sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
R,L,oc=ob.load(); A=R[key+['res','tot']+oc].copy(); A['p1']=(1/A[oc]).sum(1)/A.tot; A['win1']=(A.res.str[0]=='1').astype(float); A=A[key+['p1','win1']]
b=pd.read_sql("select hd,jcd,rno,lane,tilt,json_extract(data,'$.weight') wt,json_extract(data,'$.wind_direction') wd,json_extract(data,'$.wind_speed') ws,json_extract(data,'$.temperature') tp,json_extract(data,'$.water_temperature') wtp,json_extract(data,'$.wave_height') wv from beforeinfo where hd>='20250401'",c)
for q in ('tilt','wt','wd','ws','tp','wtp','wv'): b[q]=pd.to_numeric(b[q],errors='coerce')
b['wtd']=b.wt-b.groupby(key).wt.transform('mean'); b['tiltmax']=b.groupby(key).tilt.transform('max')
e=pd.read_sql("select hd,jcd,rno,lane,racer_no,race_name from entries where hd>='20250320'",c)
r=pd.read_sql("select hd,jcd,rno,lane,position from results where hd>='20250320'",c).merge(e,on=key+['lane']); r['pos']=pd.to_numeric(r.position,errors='coerce')
# 1日2走目: 同じ日の前のレースの着順
r=r.sort_values(['hd','racer_no','rno']); r['prev_pos']=r.groupby(['hd','racer_no']).pos.shift(1); r['nth']=r.groupby(['hd','racer_no']).cumcount()+1
# 勝負駆け: 節の得点率（前日まで。着順点 10,8,6,4,2,1）と、その節の中での順位
dd=r[['jcd','hd']].drop_duplicates().sort_values(['jcd','hd']); dd['d']=pd.to_datetime(dd.hd); dd['meet']=(dd.groupby('jcd').d.diff().dt.days!=1).groupby(dd.jcd).cumsum(); dd['day']=dd.groupby(['jcd','meet']).cumcount()+1; dd['nd']=dd.groupby(['jcd','meet']).day.transform('max')
r=r.merge(dd[['jcd','hd','meet','day','nd']],on=['jcd','hd']); r['pt']=r.pos.map({1:10,2:8,3:6,4:4,5:2,6:1}).fillna(0)
g=r.groupby(['jcd','meet','racer_no','hd']).pt.agg(['sum','count']).reset_index().sort_values('hd'); k2=['jcd','meet','racer_no']
g['cs']=g.groupby(k2)['sum'].cumsum()-g['sum']; g['cc']=g.groupby(k2)['count'].cumsum()-g['count']; g['rate']=np.where(g.cc>=2,g.cs/g.cc,np.nan); g['rrk']=g.groupby(['jcd','meet','hd']).rate.rank(ascending=False,method='min')
r=r.merge(g[k2+['hd','rate','rrk']],on=k2+['hd'],how='left')
# 予選最終日の目安: 6日開催の4日目・5日開催の3日目・4日開催の2日目（=最終日の2日前）
r['yosen_last']=(r.day==r.nd-2)&(r.nd>=4)
r['border']=np.where(r.yosen_last&r.rrk.between(13,24),'境目（13〜24位）',np.where(r.yosen_last&(r.rrk<=12),'安全圏（12位まで）',np.where(r.yosen_last&(r.rrk>24),'圏外（25位以下）','')))
X=r[key+['lane','prev_pos','nth','border','rate']]
F=b[key+['lane','tilt','wtd','wd','ws','tp','wtp','wv','tiltmax']].merge(X,on=key+['lane'],how='outer')
L=L.merge(F,on=key+['lane'],how='left'); W=L[L.win1&(L.p1>=.4)]; A=A.merge(F[F.lane==1],on=key,how='left')
def opp(nm,ser,sub=None):
    s=(W if sub is None else sub).assign(b=ser).dropna(subset=['b']); s=s[s.b!='']
    t=s.groupby('b',observed=True).agg(n=('in3','size'),mk=('mk3','mean'),ac=('in3','mean')); t['差']=t.ac-t.mk
    for nm2,q in (('前半',s[s.hd<'20260401']),('後半',s[s.hd>='20260401'])): u=q.groupby('b',observed=True).agg(mk=('mk3','mean'),ac=('in3','mean')); t[nm2]=u.ac-u.mk
    t.columns=['件数','売れ方','実際','差','前半の差','後半の差']; print(f'\n[相手] {nm}（全体の差 {W.in3.mean()-W.mk3.mean():+.3f}）'); print(t.round(3).to_string())
def one(nm,ser):
    s=A.assign(b=ser).dropna(subset=['b','p1']); s=s[s.b!='']
    t=s.groupby('b',observed=True).agg(n=('win1','size'),mk=('p1','mean'),ac=('win1','mean')); t['差']=t.ac-t.mk
    for nm2,q in (('前半',s[s.hd<'20260401']),('後半',s[s.hd>='20260401'])): u=q.groupby('b',observed=True).agg(mk=('p1','mean'),ac=('win1','mean')); t[nm2]=u.ac-u.mk
    t.columns=['件数','売れ方','実際','差','前半の差','後半の差']; print(f'\n[1号艇] {nm}（全体の差 {A.win1.mean()-A.p1.mean():+.3f}）'); print(t.round(3).to_string())
opp('チルト',pd.cut(W.tilt,[-9,-.4,.1,.6,9],labels=['-0.5','0','0.5','1.0以上'])); one('1号艇のチルト',pd.cut(A.tilt,[-9,-.4,.1,9],labels=['-0.5','0','0.5以上'])); one('だれかがチルト1.5以上',np.where(A.tiltmax>=1.5,'いる','いない'))
opp('体重（レース平均との差）',pd.cut(W.wtd,[-99,-3,-1,1,3,99],labels=['3kg以上軽い','1〜3kg軽い','ふつう','1〜3kg重い','3kg以上重い'])); one('1号艇の体重',pd.cut(A.wtd,[-99,-3,-1,1,3,99],labels=['3kg以上軽い','1〜3kg軽い','ふつう','1〜3kg重い','3kg以上重い']))
opp('その日の2走目の、1走目の着順',W.prev_pos.map({1:'1走目1着',2:'1走目2着',3:'1走目3着',4:'1走目4着',5:'1走目5着',6:'1走目6着'})); one('1号艇が2走目のときの1走目の着順',A.prev_pos.map({1:'1走目1着',2:'1走目2〜3着',3:'1走目2〜3着',4:'1走目4〜6着',5:'1走目4〜6着',6:'1走目4〜6着'}))
one('風の強さ',pd.cut(A.ws,[-1,1,3,5,7,99],labels=['0〜1m','2〜3m','4〜5m','6〜7m','8m以上'])); one('風向き（公式の16方位の番号。場ごとに意味が違うので番号のまま）×風4m以上',np.where(A.ws>=4,A.wd.fillna(-1).astype(int).astype(str),''))
one('気温',pd.cut(A.tp,[-20,5,10,15,20,25,30,50],labels=['5度以下','5〜10','10〜15','15〜20','20〜25','25〜30','30度超'])); one('波',pd.cut(A.wv,[-1,1,3,5,99],labels=['0〜1cm','2〜3cm','4〜5cm','6cm以上']))
opp('勝負駆け（予選最終日の得点率の順位）',W.border); one('1号艇の勝負駆け',A.border)
