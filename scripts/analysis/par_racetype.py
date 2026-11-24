exec(open('scripts/analysis/three_signals.py').read().split("R=lambda s,col")[0])
import sys; sys.path.insert(0,'scripts'); from common import VENUES
cut='20260201'
en=pd.read_sql("select hd,jcd,rno,lane,race_name,grade from entries where hd>='20250601'",c)
en['race_name']=en.race_name.fillna('').str.replace(r'\s+','',regex=True)
def rt(n):
    if '進入固定' in n: return '進入固定'
    for k,v in [('ドリーム','ドリーム'),('優勝戦','優勝戦'),('準優','準優'),('女子','女子'),('レディ','女子'),('ルーキ','ルーキー'),('特選','特選/特賞'),('特賞','特選/特賞'),('選抜','選抜'),('予選','予選'),('一般','一般')]:
        if k in n: return v
    return 'その他(愛称等)'
rn=en.drop_duplicates(['hd','jcd','rno'])[['hd','jcd','rno','race_name']]; rn['rtype']=rn.race_name.map(rt)
g1=en[en.lane==1][['hd','jcd','rno','grade']].rename(columns={'grade':'g1'})
na=en[en.grade=='A1'].groupby(['hd','jcd','rno']).size().rename('nA1').reset_index()
A=rn.merge(g1,on=key,how='left').merge(na,on=key,how='left'); A['nA1']=A.nA1.fillna(0).astype(int)
A['rgrp']=pd.cut(A.rno,[0,4,8,12],labels=['1-4R','5-8R','9-12R']).astype(str)
A['nA1g']=A.nA1.clip(upper=3).map({0:'A1=0',1:'A1=1',2:'A1=2',3:'A1>=3'})
A['venue']=A.jcd.map(VENUES)
print(A.rtype.value_counts().to_dict())
U=S.drop_duplicates(key+['cb']).drop_duplicates(key).merge(A,on=key,how='left')  # one ticket per race
print('union races',len(U),'ROI',U.pay.sum()/(100*len(U)))
def stat(s):
    n=len(s); 
    if n==0: return None
    h=s.pay.sort_values(ascending=False); r=lambda q:q.pay.sum()/(100*len(q)) if len(q) else float('nan')
    ex=(s.pay.sum()-h.head(3).sum())/(100*n)
    return dict(n=n,roi=r(s),n1=len(s[s.hd<cut]),r1=r(s[s.hd<cut]),n2=len(s[s.hd>=cut]),r2=r(s[s.hd>=cut]),ex3=ex)
nv=0;rows=[]
for col in ['rtype','rgrp','venue','g1','nA1g']:
    for k,s in U.groupby(col):
        st=stat(s); nv+=1; st['dim']=col; st['grp']=k; rows.append(st)
T=pd.DataFrame(rows)
pd.set_option('display.width',200); pd.set_option('display.max_rows',200)
print(T[['dim','grp','n','roi','n1','r1','n2','r2','ex3']].round(3).to_string())
print('variants',nv)
print('FAIL flags'); print(T[(T.n>=80)&(T.r1<0.8)&(T.r2<0.8)].round(3).to_string())
# Task B
o=pd.read_sql("select hd,jcd,rno,lane,lap from oriten where hd>='20250601' and lap is not null",c)
b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et from beforeinfo where hd>='20250601'",c); b['et']=pd.to_numeric(b.et,errors='coerce')
o=o.merge(b,on=key+['lane'],how='left').sort_values(key+['lap','lane']); o['r']=o.groupby(key).cumcount()+1
W=o.pivot_table(index=key,columns='r',values=['lane','et'],aggfunc='first'); W.columns=[f'{a}{b}' for a,b in W.columns]; W=W.reset_index()
W=W[(W.lane1==1)&W[['lane2','lane3']].isin([2,3]).all(1)].copy()
W['cb']=np.where(W.et2<=W.et3,'1-'+W.lane2.astype(int).astype(str)+'-'+W.lane3.astype(int).astype(str),'1-'+W.lane3.astype(int).astype(str)+'-'+W.lane2.astype(int).astype(str))
W=W.merge(p,on=key); W['pay']=np.where(W.res==W.cb,W.payout,0); W=W.merge(A,on=key,how='left')
print('\nTask B raw',stat(W))
rows=[];nb=0
for col in ['rtype','rgrp','venue','g1','nA1g']:
    for k,s in W.groupby(col):
        st=stat(s); nb+=1; st['dim']=col;st['grp']=k; rows.append(st)
for k,s in W.groupby(['rtype','rgrp']):
    st=stat(s); nb+=1; st['dim']='rtype x rno';st['grp']=k; rows.append(st)
B=pd.DataFrame(rows); print('variants',nb)
print(B[(B.n>=150)&(B.r1>1)&(B.r2>1)&(B.ex3>1)].round(3).to_string())
print(B[(B.n>=150)].sort_values('roi',ascending=False).head(8).round(3).to_string())
