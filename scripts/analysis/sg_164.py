"""SG などグレード別の 3連単 1-6-4 回収率（直近3年）。
meets.csv の hd は過去の節では最終日なので、DB の開催日を後ろへたどって節の日を決める。"""
import sqlite3,pandas as pd,datetime as dt
c=sqlite3.connect('data/boatrace.db')
M=pd.read_csv('data/meta/meets.csv',dtype=str).query("start>='20231126'")
days=pd.read_sql("select distinct jcd,hd from results where hd>='20231101'",c)
S=set(zip(days.jcd,days.hd)); rows=[]
for r in M.itertuples():
    d=dt.datetime.strptime(r.start,'%Y%m%d')
    if (r.jcd,r.start) not in S: continue
    while True:
        rows.append((r.jcd,d.strftime('%Y%m%d'),r.grade,r.title))
        p=d-dt.timedelta(1)
        if (r.jcd,p.strftime('%Y%m%d')) in S: d=p
        else: break
G=pd.DataFrame(rows,columns=['jcd','hd','grade','title']).drop_duplicates(['jcd','hd'])
p=pd.read_sql("select hd,jcd,rno,combination,payout from payouts where ticket_type='sanrensho' and hd between '20231126' and '20261125'",c)
p['combination']=p.combination.str.replace(' ','')
d=p.drop_duplicates(['hd','jcd','rno']).merge(G,on=['jcd','hd'],how='left'); d['grade']=d.grade.fillna('?')
d['hit']=d.combination=='1-6-4'; d['ret']=d.payout.where(d.hit,0); d['y']=d.hd.str[:4]
f=lambda x: pd.Series({'R':len(x),'hit':int(x.hit.sum()),'roi':round(x.ret.sum()/100/len(x),3),'top1除き':round((x.ret.sum()-x.ret.max())/100/len(x),3)})
print(d.groupby('grade').apply(f).to_string())
s=d[d.grade=='SG']
print(s.groupby('y').apply(f).to_string())
print(s[s.hit][['hd','jcd','rno','payout','title']].to_string())
print('全体',f(d).to_dict())
