"""節の何日目かで回収率が変わるか（運用者の案: 初日は荒れる・情報が少ないので3〜4日目を狙う）。3連単・締切時100倍未満。締切時オッズ。"""
import sqlite3,pandas as pd,numpy as np,warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
c=sqlite3.connect('data/boatrace.db')
dd=pd.read_sql("select distinct jcd,hd from entries where hd>='20250201'",c).sort_values(['jcd','hd']); dd['d']=pd.to_datetime(dd.hd); dd['meet']=(dd.groupby('jcd').d.diff().dt.days!=1).groupby(dd.jcd).cumsum()
dd['day']=dd.groupby(['jcd','meet']).cumcount()+1; dd['nd']=dd.groupby(['jcd','meet']).day.transform('max'); dd['last']=dd.day==dd.nd
dd['lab']=np.where(dd['last']&(dd.nd>=3),'最終日',np.where(dd.day==1,'初日',np.where(dd.day==2,'2日目',np.where(dd.day==3,'3日目',np.where(dd.day==4,'4日目','5日目以降')))))
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
O=pd.read_parquet(D+'cheap100.parquet').merge(dd[['jcd','hd','lab']],on=['jcd','hd']); O['pay']=np.where(O.hit==1,O.payout,0); O['w']=O[['a_er','b_er','c_er']].max(1)
order=['初日','2日目','3日目','4日目','5日目以降','最終日']; rows=[]
for l in order:
    s=O[O.lab==l]; n=s.groupby(['hd','jcd','rno']).ngroups; f=s[s.rk==1]; t=s[(s.w<=3)&(s.odds<20)]
    rows.append({'日':l,'レース':n,'1番人気の的中':round(f.hit.mean(),3),'1番人気の平均オッズ':round(f.odds.mean(),1),'1番人気':round(R(f),3),'6倍未満':round(R(s[s.odds<6]),3),'6〜15倍':round(R(s[(s.odds>=6)&(s.odds<15)]),3),'15〜30倍':round(R(s[(s.odds>=15)&(s.odds<30)]),3),'30〜100倍':round(R(s[s.odds>=30]),3),'3艇とも展示3位以内＆20倍未満':round(R(t),3),'同・前半':round(R(t[t.hd<'20260201']),3),'同・後半':round(R(t[t.hd>='20260201']),3)})
print(pd.DataFrame(rows).set_index('日').T.to_string())
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250301'",c).drop_duplicates(['hd','jcd','rno']).merge(dd[['jcd','hd','lab']],on=['jcd','hd'])
p['w1']=p.res.str[0]=='1'; p['man']=p.payout>=10000
g=p.groupby('lab').agg(races=('payout','size'),win1=('w1','mean'),med=('payout','median'),man=('man','mean')).loc[order]; g.columns=['レース','1号艇の1着率','払戻の中央値','万舟の割合']; print(g.round(3).T.to_string())
C=pd.read_parquet(D+'ob3.parquet').merge(dd[['jcd','hd','lab']],on=['jcd','hd']); K=C[(C.gap>=.1)&(C.xer>=5)&(C.odds<15)]
print('売れすぎ外し（売れすぎ幅0.1以上・X展示5〜6位・15倍未満）'); print(K.groupby('lab').apply(lambda s:pd.Series({'点':len(s),'回収':round(R(s),3),'前半':round(R(s[s.disc]),3),'後半':round(R(s[~s.disc]),3)})).loc[order].T.to_string())
