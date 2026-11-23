"""レースの種類（番組名）・時間帯・女子戦で分けて、1号艇の1着と売れ方の差、3連単の買い方の回収率を見る。
番組名・時間帯・出走メンバーは前夜に分かる情報なので、選び方そのものに後出しは無い（払戻は確定）。2025-06〜、前半〜2026-01／後半2026-02〜。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'; key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db'); cut='20260201'
e=pd.read_sql("select hd,jcd,rno,race_name,close_time, sum(case when json_extract(data,'$.grade') in ('A1') then 1 else 0 end) a1 from entries where hd>='20250601' group by 1,2,3",c)
sx=pd.read_sql("select hd,jcd,rno,lane,racer_no from entries where hd>='20250601'",c); rs=pd.read_sql("select racer_no,sex from (select racer_no,json_extract(data,'$.sex') sex, max(hd) from racer_stats group by racer_no)",c)
sx['racer_no']=sx.racer_no.astype(str); rs['racer_no']=rs.racer_no.astype(str); sx=sx.merge(rs,on='racer_no',how='left'); fem=sx.groupby(key).sex.apply(lambda s:(s.astype(str)=='2').sum()).rename('nf').reset_index()
e=e.merge(fem,on=key,how='left'); n=e.race_name.fillna('')
def typ(r):
    nm=r.race_name or ''
    if '進入固定' in nm: return '進入固定'
    if '優勝' in nm and '準' not in nm: return '優勝戦'
    if '準優' in nm: return '準優勝戦'
    if 'ドリーム' in nm: return 'ドリーム戦'
    if '選抜' in nm or '特選' in nm or '特賞' in nm: return '選抜・特選・特賞'
    if '予選' in nm: return '予選'
    if '一般' in nm: return '一般'
    return 'その他（場ごとの名前）'
e['typ']=e.apply(typ,axis=1); e['hr']=e.close_time.str[:2].astype(int); e['tz']=np.where(e.hr<11,'朝（〜10時台）',np.where(e.hr<16,'昼（11〜15時台）',np.where(e.hr<18,'夕（16〜17時台）','ナイター（18時〜）')))
e['women']=np.where(e.nf>=6,'女子戦（6人とも女子）',np.where(e.nf>=1,'男女混合','男子のみ'))
O=pd.read_parquet(D+'cheap100.parquet'); O=O[O.hd>='20250601']
ph1=O[O.a_lane==1].groupby(key).q.sum().rename('p1').reset_index()   # 100倍未満の①頭の組だけなので概算
p=pd.read_sql("select hd,jcd,rno,combination res from payouts where ticket_type='sanrensho' and hd>='20250601'",c).drop_duplicates(key); p['w1']=(p.res.str[0]=='1').astype(float)
A=e.merge(p,on=key).merge(ph1,on=key,how='left')
o=pd.read_sql("select hd,jcd,rno,lane,lap from oriten",c); o['lap']=o.groupby(key).lap.rank(method='min')
B=O.copy()
for q in 'abc': B=B.merge(o.rename(columns={'lane':f'{q}_lane','lap':f'{q}_lap'}),on=key+[f'{q}_lane'],how='inner')
B['pay']=np.where(B.hit==1,B.payout,0); B=B.merge(e[key+['typ','tz','women']],on=key)
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
for col in ('typ','tz','women'):
    rows=[]
    for g,s in A.groupby(col):
        b=B[B[col]==g]; box=b[b[['a_lap','b_lap','c_lap']].max(1)<=3]; fav=b[b.rk==1]; h1=box[box.a_lane==1]
        rows.append({'区分':g,'レース':len(s),'①1着':round(s.w1.mean(),3),'①頭の売れ方(概算)':round(s.p1.mean(),3),
            '一周上位3艇ボックス':round(R(box),3),'同 前半':round(R(box[box.hd<cut]),3),'同 後半':round(R(box[box.hd>=cut]),3),
            '①頭の一周上位ボックス':round(R(h1),3),'1番人気1点':round(R(fav),3),'1番人気 前半':round(R(fav[fav.hd<cut]),3),'1番人気 後半':round(R(fav[fav.hd>=cut]),3)})
    print('\n=== ',col); print(pd.DataFrame(rows).set_index('区分').to_string())
print('\n=== 進入固定を掘る（前夜に分かる）')
K=B[B.typ=='進入固定']; 
def rep(nm,s): mo=s.groupby(s.hd.str[:6]).apply(R); print(f'{nm} | 点 {len(s)} レース {s.groupby(key).ngroups} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
rep('全部（100倍未満）',K); rep('①頭',K[K.a_lane==1]); rep('①頭・一周上位3艇',K[(K.a_lane==1)&(K[['a_lap','b_lap','c_lap']].max(1)<=3)]); rep('①頭・①一周1位',K[(K.a_lane==1)&(K.a_lap==1)])
rep('①頭・20倍未満',K[(K.a_lane==1)&(K.odds<20)]); rep('①頭・1-2-3/1-3-2以外の①頭20倍未満',K[(K.a_lane==1)&(K.odds<20)&~K.cb.isin(['1-2-3','1-3-2'])])
for lo,hi in ((0,10),(10,20),(20,50),(50,100)): rep(f'①頭 {lo}〜{hi}倍',K[(K.a_lane==1)&(K.odds>=lo)&(K.odds<hi)])
