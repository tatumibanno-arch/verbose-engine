"""夜通しの総当たり（3連単・締切時100倍未満の組、2025-06〜2026-11）。運用者「無限に検証回し続けて、明日の朝までに見つけてきて」。
条件の部品（1〜3着に置く艇それぞれの 一周・まわり足・直線の順位、公式展示順位、モーター差、体重差、艇番、選手の型（まくり・差し）、F持ち、
その頭の売れ方、オッズ帯、人気順、風、波、①の一周順位、節の日）を最大3つ組み合わせて全部試す。
偶然を見分けるため:
  1) 期間を3つに分ける: 探す 2025-06〜12 ／ 確かめる 2026-01〜06 ／ 最後 2026-07〜11（最後は全部終わってから1回だけ見る）
  2) 探す期間で1.05以上・点数1,500以上のものだけ次へ
  3) 「売れ方どおりに当たりを引き直した世界」（どの条件も本当は0.75の世界）でも同じ手順を回し、残る本数を比べる
使い方: python overnight_hunt.py build ／ python overnight_hunt.py mine real|fake1|fake2|fake3 ／ python overnight_hunt.py report"""
import sqlite3, sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'; key=['hd','jcd','rno']
def build():
    c=sqlite3.connect('data/boatrace.db')
    O=pd.read_parquet(D+'cheap100.parquet'); O=O[O.hd>='20250601']
    o=pd.read_sql("select hd,jcd,rno,lane,lap,mawari,chokusen from oriten",c)
    for q in ('lap','mawari','chokusen'): o[q]=o.groupby(key)[q].rank(method='min')
    b=pd.read_sql("select hd,jcd,rno,lane,json_extract(data,'$.weight') wt,json_extract(data,'$.wind_speed') ws,json_extract(data,'$.wave_height') wv from beforeinfo where hd>='20250601'",c)
    for q in ('wt','ws','wv'): b[q]=pd.to_numeric(b[q],errors='coerce')
    b['wtd']=b.wt-b.groupby(key).wt.transform('mean'); env=b.groupby(key)[['ws','wv']].max().reset_index()
    # 選手の型（style_cal.py と同じ作り方）
    K=pd.concat([pd.read_parquet(f'data/parquet/kimarite_{y}.parquet') for y in (2024,2025,2026)]); K['jcd']=K.jcd.astype(int).astype(str).str.zfill(2); K['rno']=K.rno.astype(int)
    r=pd.read_sql("select r.hd,r.jcd,r.rno,r.lane,r.position, cast(json_extract(r.data,'$.entrance_position') as int) crs, e.racer_no from results r join entries e using(hd,jcd,rno,lane) where r.hd>='20240501'",c)
    r['racer_no']=r.racer_no.astype(int); r=r.merge(K[key+['kimarite']],on=key,how='left'); r['ym']=r.hd.str[:6]
    out=r[r.crs.between(2,6)]; w=out[out.position==1]; m=out.groupby(['racer_no','ym']).size().rename('st').to_frame()
    for kk,nm in (('まくり','mk'),('まくり差し','ms'),('差し','sa')): m[nm]=w[w.kimarite==kk].groupby(['racer_no','ym']).size()
    m=m.fillna(0).reset_index(); months=sorted(r.ym.unique()); rows=[]
    for i,ym in enumerate(months):
        if i<12: continue
        s=m[m.ym.isin(months[i-12:i])].groupby('racer_no')[['st','mk','ms','sa']].sum(); s['ym']=ym; rows.append(s.reset_index())
    S=pd.concat(rows); S['mkr']=(S.mk+S.ms)/(S.st+10); S['sar']=S.sa/(S.st+10)
    e=pd.read_sql("select hd,jcd,rno,lane,racer_no from entries where hd>='20250601'",c); e['racer_no']=e.racer_no.astype(int); e['ym']=e.hd.str[:6]
    e=e.merge(S[['racer_no','ym','mkr','sar']],on=['racer_no','ym'],how='left')
    rc=pd.read_parquet(D+'result_codes.parquet'); rc['d']=pd.to_datetime(rc.hd)
    rc['term']=np.where(rc.d.dt.month<=4,(rc.d.dt.year-1).astype(str)+'B',np.where(rc.d.dt.month<=10,rc.d.dt.year.astype(str)+'A',rc.d.dt.year.astype(str)+'B'))
    f=rc[rc.code=='F'][['racer_no','hd','term']]; e['d']=pd.to_datetime(e.hd); e['term']=np.where(e.d.dt.month<=4,(e.d.dt.year-1).astype(str)+'B',np.where(e.d.dt.month<=10,e.d.dt.year.astype(str)+'A',e.d.dt.year.astype(str)+'B'))
    mm=e[['racer_no','term','hd']].merge(f.rename(columns={'hd':'fhd'}),on=['racer_no','term']); mm=mm[mm.fhd<mm.hd].groupby(['racer_no','term','hd']).size().rename('fn').reset_index()
    e=e.merge(mm,on=['racer_no','term','hd'],how='left'); e['fn']=e.fn.fillna(0)
    P=o.merge(b[key+['lane','wtd']],on=key+['lane'],how='left').merge(e[key+['lane','mkr','sar','fn']],on=key+['lane'],how='left')
    O=O.merge(o[key].drop_duplicates(),on=key)
    for p in 'abc': O=O.merge(P.rename(columns={'lane':f'{p}_lane','lap':f'{p}_lap','mawari':f'{p}_maw','chokusen':f'{p}_cho','wtd':f'{p}_wtd','mkr':f'{p}_mkr','sar':f'{p}_sar','fn':f'{p}_fn'}),on=key+[f'{p}_lane'],how='left')
    O=O.merge(env,on=key,how='left').merge(o[o.lane==1][key+['lap']].rename(columns={'lap':'lap1'}),on=key,how='left')
    dd=pd.read_sql("select distinct jcd,hd from entries where hd>='20250401'",c).sort_values(['jcd','hd']); dd['dt']=pd.to_datetime(dd.hd); dd['meet']=(dd.groupby('jcd').dt.diff().dt.days!=1).groupby(dd.jcd).cumsum(); dd['day']=dd.groupby(['jcd','meet']).cumcount()+1
    O=O.merge(dd[['jcd','hd','day']],on=['jcd','hd'],how='left')
    O['pay']=np.where(O.hit==1,O.payout,0).astype('float32'); O['per']=np.where(O.hd<'20260101',0,np.where(O.hd<'20260701',1,2)).astype('int8')
    qs={col:O[col].quantile([1/3,2/3]).values for col in ['a_mkr','a_sar']}
    A={}
    for p,nm in (('a','1着'),('b','2着'),('c','3着')):
        for k in range(1,7): A[f'{nm}={k}号艇']=O[f'{p}_lane']==k
        for q,qn in (('lap','一周'),('maw','まわり足'),('cho','直線')):
            A[f'{nm}{qn}1位']=O[f'{p}_{q}']==1; A[f'{nm}{qn}1〜2位']=O[f'{p}_{q}']<=2; A[f'{nm}{qn}3位以内']=O[f'{p}_{q}']<=3; A[f'{nm}{qn}5〜6位']=O[f'{p}_{q}']>=5
        A[f'{nm}公式展示1〜2位']=O[f'{p}_er']<=2; A[f'{nm}公式展示5〜6位']=O[f'{p}_er']>=5
        A[f'{nm}モーター+5以上']=O[f'{p}_mod']>=5; A[f'{nm}モーター-5以下']=O[f'{p}_mod']<=-5
        A[f'{nm}体重2kg以上軽い']=O[f'{p}_wtd']<=-2; A[f'{nm}体重2kg以上重い']=O[f'{p}_wtd']>=2
        A[f'{nm}まくり型']=O[f'{p}_mkr']>=qs['a_mkr'][1]; A[f'{nm}差し型']=O[f'{p}_sar']>=qs['a_sar'][1]; A[f'{nm}F持ち']=O[f'{p}_fn']>=1
        A[f'{nm}A級']=O[f'{p}_g']>=3; A[f'{nm}B級']=O[f'{p}_g']<=2
    for lo,hi in ((0,10),(10,20),(20,40),(40,100)): A[f'オッズ{lo}〜{hi}倍']=(O.odds>=lo)&(O.odds<hi)
    for lo,hi in ((0,.3),(.3,.5),(.5,.7),(.7,1.01)): A[f'頭の売れ方{int(lo*100)}〜{int(hi*100)}%']=(O.ph>=lo)&(O.ph<hi)
    A['人気3番以内']=O.rk<=3; A['人気4〜10番']=O.rk.between(4,10); A['人気11番以下']=O.rk>=11
    A['風0〜2m']=O.ws<=2; A['風5m以上']=O.ws>=5; A['波0〜1cm']=O.wv<=1; A['波4cm以上']=O.wv>=4
    A['①一周1位']=O.lap1==1; A['①一周4〜6位']=O.lap1>=4; A['初日']=O.day==1; A['4日目以降']=O.day>=4; A['1〜4R']=O.rno<=4; A['9〜12R']=O.rno>=9
    names=list(A); M=np.column_stack([A[n].fillna(False).values for n in names]).astype(bool)
    np.save(D+'oh_M.npy',M); json.dump(names,open(D+'oh_names.json','w'),ensure_ascii=False)
    O[key+['cb','odds','inv','tot','hit','payout','pay','per']].to_parquet(D+'oh_rows.parquet'); print('組',len(O),'部品',len(names))
def fake_pay(seed):
    R=pd.read_parquet(D+'oh_rows.parquet'); rng=np.random.default_rng(seed); rid=(R.hd+R.jcd+R.rno.astype(str)).values; gi,_=pd.factorize(rid); ng=gi.max()+1
    q=(R.inv/R.tot).values; sq=np.bincount(gi,weights=q,minlength=ng); key_=np.log(rng.random(len(R)))/R.inv.values; mx=np.full(ng,-np.inf); np.maximum.at(mx,gi,key_)
    win=key_==mx[gi]; has=(rng.random(ng)<sq)[gi]; return np.where(win&has,R.odds.values*100,0).astype('float32')
def mine(tag):
    M=np.load(D+'oh_M.npy'); names=json.load(open(D+'oh_names.json')); R=pd.read_parquet(D+'oh_rows.parquet'); per=R.per.values
    pay=R.pay.values if tag=='real' else fake_pay({'fake1':1,'fake2':2,'fake3':3}[tag])
    S0=M[per==0]; Y0=pay[per==0]; F0=S0.astype(np.float32); n=len(names); MIN=1500; found=[]
    for i in range(n):
        mi=S0[:,i]
        if mi.sum()<MIN: continue
        for j in range(i+1,n):
            mij=mi&S0[:,j]; cnt2=int(mij.sum())
            if cnt2<MIN: continue
            y=Y0[mij]; r2=y.sum()/(100*cnt2)
            if r2>=1.05: found.append(((i,j),cnt2,r2))
            sub=F0[mij]; cnt=sub.sum(0); sm=y@sub
            for k in range(j+1,n):
                if MIN<=cnt[k]<cnt2:
                    r3=sm[k]/(100*cnt[k])
                    if r3>=1.05: found.append(((i,j,k),int(cnt[k]),float(r3)))
        if i%10==0: print(tag,i,'/',n,'見つかった',len(found),flush=True)
    rows=[]
    for idx,c0,r0 in found:
        m=np.logical_and.reduce([M[:,t] for t in idx]); row=dict(条件=' ＆ '.join(names[t] for t in idx),探す点数=c0,探す=round(r0,3))
        for p,nm in ((1,'確かめる'),(2,'最後')):
            mm=m&(per==p); cc=int(mm.sum()); row[nm+'点数']=cc; row[nm]=round(pay[mm].sum()/(100*cc),3) if cc else np.nan
        rows.append(row)
    T=pd.DataFrame(rows); T.to_csv(D+f'oh_{tag}.csv',index=False)
    print(f'[{tag}] 探す期間で1.05以上: {len(T)}本 → 確かめる期間でも1.0以上: {(T.確かめる>=1).sum() if len(T) else 0}本 → 最後も1.0以上: {((T.確かめる>=1)&(T.最後>=1)).sum() if len(T) else 0}本',flush=True)
if __name__=='__main__':
    a=sys.argv[1]
    if a=='build': build()
    elif a=='mine': mine(sys.argv[2])
