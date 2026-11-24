"""par_meet_run: 勝負駆け・部品交換・体重の15仮説。オッズ不使用。ROI=払戻/購入（確定払戻、1点100円）。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/'; key=['hd','jcd','rno']; cut='20260201'
X=pd.read_pickle(S+'X.pkl'); p=pd.read_pickle(S+'p.pkl')
X=X[X.position.notna()|True]
# 前回体重（その選手の直前の別日の beforeinfo 体重）
X=X.sort_values(['racer_no','hd','jcd','rno']); X['wtot']=X.wt+X.adj.fillna(0)
prev=X.dropna(subset=['wt']).groupby(['racer_no','hd']).wt.mean().reset_index().sort_values(['racer_no','hd'])
prev['pw']=prev.groupby('racer_no').wt.shift(1); prev['pd']=pd.to_datetime(prev.hd); prev['pdd']=prev.groupby('racer_no').pd.shift(1)
prev['gapd']=(prev.pd-prev.pdd).dt.days
X=X.merge(prev[['racer_no','hd','pw','gapd']],on=['racer_no','hd'],how='left'); X['dw']=X.wt-X.pw; X.loc[X.gapd>60,'dw']=np.nan  # 2か月超の空きは除外
X=X.sort_values(key+['lane'])
def W(col): 
    t=X.pivot_table(index=key,columns='lane',values=col,aggfunc='first'); t.columns=[f'{col}{i}' for i in t.columns]; return t
parts=X.assign(pc=(X.parts.fillna('')!='')|X.prop.notna()).pivot_table(index=key,columns='lane',values='pc',aggfunc='max'); parts.columns=[f'pc{i}' for i in parts.columns]
R=pd.concat([W('gap'),W('nw'),W('wtot'),W('dw'),W('wave_height'),W('wind_speed'),parts],axis=1).reset_index()
lastr=X.groupby(key)['last'].max().rename('last').reset_index(); R=R.merge(lastr,on=key).merge(p,on=key)
R['nok']=R[[f'wtot{i}' for i in range(1,7)]].notna().all(1)
n=len(R); print('races',n,'last-day races',int(R['last'].sum()))
def T(*a): return a
g=lambda i: R[f'gap{i}']; 
def tick(a,b,c): return a+'-'+b+'-'+c
def bub(lo=-6,hi=6):
    # 2-4枠のうち gap が [lo,hi] の艇をちょうど1艇 → その枠
    m=np.stack([(g(i)>=lo)&(g(i)<=hi) for i in (2,3,4)],1); one=m.sum(1)==1
    return np.where(one,np.array(['2','3','4'])[m.argmax(1)],'')
def others(b):  # 2-4のb以外2枠
    o1=np.where(b=='2','3','2'); o2=np.where(b=='4','3','4'); return o1,o2
H={}
F4=lambda mask: (mask,[('2','1','3'),('3','1','2'),('2','3','1'),('3','2','1')])
A2=lambda mask: (mask,[('1','2','3'),('1','3','2')])
L=R['last']
H['H1 ①圏外(gap<=-8) 最終日→②③頭の①2着3着(4点)']=F4(L&(g(1)<=-8))
H['H2 ①当確(gap>=8) 最終日→1-2-3,1-3-2']=A2(L&(g(1)>=8))
b=bub(); o1,o2=others(b); hasb=L&(b!='')
H['H3 ①当確&2-4枠にボーダー1艇b→1-b-x(2点)']=(hasb&(g(1)>=8),[('1',b,o1),('1',b,o2)])
H['H4 同・①条件なし→1-b-x(2点)']=(hasb,[('1',b,o1),('1',b,o2)])
H['H5 同→b-1-x(2点)']=(hasb,[(b,'1',o1),(b,'1',o2)])
H['H6 ①ボーダー(|gap|<=6)→1-2-3,1-3-2']=A2(L&(g(1).abs()<=6))
H['H7 ①圏外&②ボーダー→2-1-3,2-1-4']=(L&(g(1)<=-8)&(g(2).abs()<=6),[('2','1','3'),('2','1','4')])
pc=lambda i: R[f'pc{i}'].fillna(False).astype(bool)
oth=lambda: pc(2)|pc(3)|pc(4)|pc(5)|pc(6)
H['H8 ①部品交換あり→②③頭の①2着3着(4点)']=F4(pc(1))
H['H9 ①部品交換あり→1-2-3,1-3-2']=A2(pc(1))
bp=np.stack([pc(2)&~pc(3)&~pc(4),pc(3)&~pc(2)&~pc(4),pc(4)&~pc(2)&~pc(3)],1); one=bp.any(1)&~pc(1)
bb=np.where(one,np.array(['2','3','4'])[bp.argmax(1)],''); p1,p2=others(bb)
H['H10 2-4枠の1艇だけ部品交換(①なし)→b-1-x(2点)']=(one,[(bb,'1',p1),(bb,'1',p2)])
H['H11 同→1-b-x(2点)']=(one,[('1',bb,p1),('1',bb,p2)])
dif=R.wtot1-R[[f'wtot{i}' for i in range(2,7)]].mean(1); rough=(R.wave_height1>=5)|(R.wind_speed1>=5)
H['H12 ①が他より3kg以上軽い→1-2-3,1-3-2']=A2(R.nok&(dif<=-3))
H['H13 ①が他より2kg以上重い&荒れ(波5以上or風5以上)→②③頭(4点)']=F4(R.nok&(dif>=2)&rough)
H['H14 ①体重が前回比+1.5kg以上→②③頭(4点)']=F4(R.dw1>=1.5)
H['H15 ①体重が前回比-1.5kg以下→1-2-3,1-3-2']=A2(R.dw1<=-1.5)
def ev(mask,tk,sub=None):
    s=R[mask.values if hasattr(mask,'values') else mask].copy() if sub is None else sub
    idx=np.where(mask)[0]; s=R.iloc[idx]
    pay=np.zeros(len(s)); stake=np.zeros(len(s))
    for t in tk:
        a=[ (x[idx] if isinstance(x,np.ndarray) else x) for x in t]
        cb=a[0]+'-'+a[1]+'-'+a[2] if not all(isinstance(x,str) for x in a) else '-'.join(a)
        cb=pd.Series(np.char.add(np.char.add(np.asarray(a[0],dtype=str),'-'),np.char.add(np.char.add(np.asarray(a[1],dtype=str),'-'),np.asarray(a[2],dtype=str))) if not all(isinstance(x,str) for x in a) else '-'.join(a)).values if False else cb
        cb=np.asarray(cb) if not isinstance(cb,str) else np.full(len(s),cb)
        pay+=np.where(s.res.values==cb,s.payout.values,0); stake+=100
    return s,pay,stake
rows=[]
for k,(mask,tk) in H.items():
    mask=np.asarray(mask,dtype=bool)
    if mask.sum()==0: rows.append((k,0)); continue
    s,pay,st=ev(mask,tk)
    hd=s.hd.values; roi=lambda m: pay[m].sum()/st[m].sum() if m.sum() else np.nan
    o=np.argsort(-pay); ex1=np.ones(len(pay),bool); ex1[o[:1]]=False; ex3=np.ones(len(pay),bool); ex3[o[:3]]=False
    mo=pd.DataFrame({'m':pd.Series(hd).str[:6],'p':pay,'s':st}).groupby('m').sum(); mr=mo.p/mo.s
    # ベースライン：同じ点数の買い方を全レースで買った回収（①絡みの固定買い）
    rows.append((k,len(s),(pay>0).mean(),roi(np.ones(len(pay),bool)),roi(hd<cut),roi(hd>=cut),roi(ex1),roi(ex3),f'{(mr>1).sum()}/{len(mr)}',pay.max()/pay.sum() if pay.sum() else np.nan))
df=pd.DataFrame(rows,columns=['hypothesis','n','hit','ROI','前半','後半','除1位','除上位3','勝ち月','最大1発の比率'])
pd.set_option('display.width',250,'display.max_colwidth',60); print(df.round(3).to_string())
# ベースライン
def base(tk):
    pay=np.zeros(n)
    for t in tk: pay+=np.where(R.res.values==t,R.payout.values,0)
    return pay.sum()/(100*len(tk)*n)
print('全レース 1-2-3,1-3-2:',round(base(['1-2-3','1-3-2']),3),' 2-1-3,3-1-2,2-3-1,3-2-1:',round(base(['2-1-3','3-1-2','2-3-1','3-2-1']),3))
Rl=R[R['last']]; 
for nm,tk in (('1-2-3,1-3-2',['1-2-3','1-3-2']),('F4',['2-1-3','3-1-2','2-3-1','3-2-1'])):
    pay=sum(np.where(Rl.res.values==t,Rl.payout.values,0) for t in tk); print('予選最終日全体',nm,round(pay.sum()/(100*len(tk)*len(Rl)),3),len(Rl))
print('\n== H7/H13 感度（探索。試行数に加算）')
def q(nm,mask,tk):
    mask=np.asarray(mask,dtype=bool); s,pay,st=ev(mask,tk); hd=s.hd.values
    r=lambda m: round(pay[m].sum()/st[m].sum(),3) if m.sum() else np.nan
    o=np.argsort(-pay); ex=np.ones(len(pay),bool); ex[o[:5]]=False
    mo=pd.DataFrame({'m':pd.Series(hd).str[:6],'p':pay,'s':st}).groupby('m').sum(); mr=mo.p/mo.s
    print(f'{nm:50s} n={len(s)} hit={(pay>0).mean():.3f} ROI={r(np.ones(len(pay),bool))} 前{r(hd<cut)} 後{r(hd>=cut)} 除上位5={r(ex)} 勝ち月{(mr>1).sum()}/{len(mr)}')
for lo in (-6,-8,-12):
    for bw in (3,6,9):
        q(f'①gap<={lo} & |②gap|<={bw} 2-1-3,2-1-4',L&(g(1)<=lo)&(g(2).abs()<=bw),[('2','1','3'),('2','1','4')])
q('①gap<=-8 & ②gap>=-6 (②がボーダー以上) 2-1-3,2-1-4',L&(g(1)<=-8)&(g(2)>=-6),[('2','1','3'),('2','1','4')])
q('①gap<=-8 & |②gap|<=6  2-1-3のみ',L&(g(1)<=-8)&(g(2).abs()<=6),[('2','1','3')])
q('①gap<=-8 & |②gap|<=6  2-1-x全4(3,4,5,6)',L&(g(1)<=-8)&(g(2).abs()<=6),[('2','1','3'),('2','1','4'),('2','1','5'),('2','1','6')])
q('①gap<=-8 全体 2-1-3,2-1-4(②条件なし)',L&(g(1)<=-8),[('2','1','3'),('2','1','4')])
q('H13: 荒れ&①重い 1-2-3,1-3-2 (逆)',R.nok&(dif>=2)&rough,[('1','2','3'),('1','3','2')])
q('H13 閾値 ①2kg重い 荒れなし→②③頭(4点)',R.nok&(dif>=2)&~rough,[('2','1','3'),('3','1','2'),('2','3','1'),('3','2','1')])
q('全レース 2-1-3,2-1-4（基準）',np.ones(n,bool),[('2','1','3'),('2','1','4')])
