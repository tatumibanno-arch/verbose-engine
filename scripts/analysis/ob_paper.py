"""紙上の候補「売れすぎ外し×安い組」の採点（お金は使わない）。条件は notes/decisions.md 2026-11-23 の固定どおり。
  ob-cheap-v1  : 3連単。A=①頭70%以上、B=①頭80%以上（BはAの内数）
  ob-cheap2-v1 : 2連単。①頭60%以上
    python scripts/analysis/ob_paper.py                       # 2026-11-24 以降を採点して notes/bets/ob-paper.md を作り直す
    python scripts/analysis/ob_paper.py --from 20261001 --to 20261031 --out /tmp/x.md   # 動作確認用
判定は締切時オッズ（締切前に見えるオッズとは違い、実際より甘い）。見立ての係数は data/models/ob_cheap_v1.json に固定。
成績・3連単と2連単の締切時オッズ・直前情報が data/boatrace.db に入った後に呼ぶ。何度呼んでも同じ結果。"""
import argparse, json, sqlite3, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent.parent.parent; sys.path.insert(0,str(ROOT/'scripts'))
from common import VENUES  # noqa: E402
START='20261124'; KEY=['hd','jcd','rno']; M=json.load(open(ROOT/'data/models/ob_cheap_v1.json'))
CAP=15.0; GAP3=.10; GAP2=.08; MO=-5.0

def lanes(c,a,b):
    e=pd.read_sql(f"select hd,jcd,rno,lane,racer_no,json_extract(data,'$.national_win_rate') nw,json_extract(data,'$.motor_2ren_rate') mo from entries where hd between '{a}' and '{b}'",c)
    if e.empty: return None
    for q in ('nw','mo'): e[q]=pd.to_numeric(e[q],errors='coerce')
    e['racer_no']=e.racer_no.astype(str)
    s=pd.read_sql("select hd shd,racer_no,data from racer_stats where hd>='20240401'",c); j=s.data.apply(json.loads)
    for k in range(1,7):
        s[f'n{k}']=j.apply(lambda x:x.get(f'c{k}_starts') or 0); s[f'a{k}']=j.apply(lambda x:(x.get(f'c{k}_pos1') or 0)+(x.get(f'c{k}_pos2') or 0)); s[f'b{k}']=j.apply(lambda x:(x.get(f'c{k}_pos3') or 0))
    s['racer_no']=s.racer_no.astype(str); s=s.drop(columns='data'); e['d']=pd.to_datetime(e.hd); s['d']=pd.to_datetime(s.shd)
    e=pd.merge_asof(e.sort_values('d'),s.sort_values('d'),on='d',by='racer_no',allow_exact_matches=False)
    pick=lambda p:np.array([getattr(r,f'{p}{r.lane}') for r in e.itertuples()],dtype=float)
    e['cn']=np.nan_to_num(pick('n')); e['c2']=np.nan_to_num(pick('a')); e['c3']=e.c2+np.nan_to_num(pick('b'))
    bi=pd.read_sql(f"select hd,jcd,rno,lane,exhibition_time et from beforeinfo where hd between '{a}' and '{b}'",c); bi['et']=pd.to_numeric(bi.et,errors='coerce'); bi['er']=bi.groupby(KEY).et.rank(method='min')
    return e[KEY+['lane','nw','mo','cn','c2','c3']].merge(bi[KEY+['lane','er']],on=KEY+['lane'],how='left')

def sig(z): return 1/(1+np.exp(-z))

def run(c,a,b):
    E=lanes(c,a,b); bets=[]
    if E is None: return pd.DataFrame()
    pay={t:pd.read_sql(f"select hd,jcd,rno,combination res,payout from payouts where ticket_type='{t}' and hd between '{a}' and '{b}'",c).drop_duplicates(KEY).set_index(KEY) for t in ('sanrensho','nirensho')}
    # ---- 3連単
    o=pd.read_sql(f"select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd between '{a}' and '{b}' and odds>0",c)
    if len(o):
        o['inv']=1/o.odds; t=o.groupby(KEY).inv.agg(tot='sum',n='size'); h=o[o.cb.str[0]=='1']; hn=h.groupby(KEY).inv.agg(h1='sum',hn='size'); t=t.join(hn); t=t[(t.n>=110)&(t.hn==20)]; t['p1']=t.h1/t.tot
        m=M['t3']; w=m['coef']; od=h.set_index(KEY+['cb']).odds
        for k,g in E[E.lane>=2].dropna(subset=['mo','nw']).groupby(KEY):
            if k not in t.index or len(g)<5 or t.loc[k,'p1']<.7: continue
            hh=h[(h.hd==k[0])&(h.jcd==k[1])&(h.rno==k[2])]; g=g.copy()
            g['mk']=[hh[hh.cb.str[2:].str.contains(str(l))].inv.sum()/t.loc[k,'h1'] for l in g.lane]
            g['r3']=(g.c3+10*g.lane.astype(str).map(m['base3']))/(g.cn+10); g['r2']=(g.c2+10*g.lane.astype(str).map(m['base2']))/(g.cn+10)
            z=m['intercept']+w[0]*g.r3+w[1]*g.r2+w[2]*g.mo/100+w[3]*g.nw/10+sum(w[4+i]*(g.lane==3+i) for i in range(4)); g['f']=sig(z); g['gap']=g.mk-g.f
            g=g.sort_values('gap',ascending=False,kind='stable'); x=g.iloc[0]; rest=g.iloc[1:].sort_values('f',ascending=False,kind='stable').head(3)
            if not (x.gap>=GAP3 and x.mo-rest.mo.mean()<=MO and x.er>=5 and rest.er.mean()<x.er): continue
            for p,q in itertools.permutations(sorted(rest.lane),2):
                cb=f'1-{p}-{q}'; ov=float(od.loc[k+(cb,)])
                if ov<CAP: bets.append(('ob-cheap-v1',)+k+(cb,ov,float(t.loc[k,'p1']),int(x.lane),float(x.gap),float(x.mo-rest.mo.mean()),float(x.er)))
    # ---- 2連単
    o=pd.read_sql(f"select hd,jcd,rno,combination cb,odds from odds where ticket_type='nirensho' and hd between '{a}' and '{b}' and odds>0",c)
    if len(o):
        o['inv']=1/o.odds; t=o.groupby(KEY).inv.agg(tot='sum',n='size'); h=o[o.cb.str[0]=='1'].copy(); h['lane']=h.cb.str[2].astype(int); t=t.join(h.groupby(KEY).inv.sum().rename('h1')); t=t[t.n>=28]; t['p1']=t.h1/t.tot
        m=M['t2']; w=m['coef']; h=h.merge(E,on=KEY+['lane'],how='left').dropna(subset=['mo','nw','er'])
        for k,g in h.groupby(KEY):
            if k not in t.index or t.loc[k,'p1']<.6 or len(g)<2: continue
            g=g.copy(); g['mk']=g.inv/t.loc[k,'h1']; g['r2']=(g.c2+10*g.lane.astype(str).map(m['base2']))/(g.cn+10)
            z=m['intercept']+w[0]*g.r2+w[1]*g.mo/100+w[2]*g.nw/10+sum(w[3+i]*(g.lane==3+i) for i in range(4)); g['f']=sig(z); g['gap']=g.mk-g.f
            g=g.sort_values('gap',ascending=False,kind='stable'); x=g.iloc[0]; rest=g.iloc[1:]
            if not (x.gap>=GAP2 and x.mo-rest.mo.mean()<=MO and x.er>=5 and rest.er.mean()<x.er): continue
            for r in rest.itertuples():
                if r.odds<CAP: bets.append(('ob-cheap2-v1',)+k+(r.cb,float(r.odds),float(t.loc[k,'p1']),int(x.lane),float(x.gap),float(x.mo-rest.mo.mean()),float(x.er)))
    B=pd.DataFrame(bets,columns=['rule']+KEY+['cb','odds','p1','X','gap','modiff','xer'])
    if B.empty: return B
    res=[];pp=[]
    for r in B.itertuples():
        P=pay['sanrensho' if r.rule=='ob-cheap-v1' else 'nirensho']; k=(r.hd,r.jcd,r.rno)
        if k in P.index: res.append(P.loc[k,'res']); pp.append(int(P.loc[k,'payout']) if P.loc[k,'res']==r.cb else 0)
        else: res.append(None); pp.append(np.nan)
    B['res']=res; B['pay']=pp; return B

def table(s):
    s=s.dropna(subset=['pay']); 
    if s.empty: return '（まだ該当なし）\n'
    out='| 月 | レース | 点数 | 的中レース | 購入 | 払戻 | 回収率 |\n|---|---|---|---|---|---|---|\n'; s=s.assign(ym=s.hd.str[:6],rid=s.hd+s.jcd+s.rno.astype(str))
    for nm,g in list(s.groupby('ym'))+[('通算',s)]:
        out+=f"| {nm} | {g.rid.nunique()} | {len(g)} | {g[g.pay>0].rid.nunique()} | {len(g)*100:,} | {int(g.pay.sum()):,} | {g.pay.sum()/(len(g)*100):.2f} |\n"
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--from',dest='a',default=START); ap.add_argument('--to',dest='b',default='20991231'); ap.add_argument('--out',default=str(ROOT/'notes/bets/ob-paper.md')); A=ap.parse_args()
    c=sqlite3.connect(ROOT/'data/boatrace.db'); B=run(c,A.a,A.b)
    md=f"# 紙上の候補「売れすぎ外し×安い組」（{A.a}〜）\n\n条件は notes/decisions.md 2026-11-23 の固定どおり。お金は使わない。各点100円として計算。\n判定は締切時オッズ＝実際より甘い。合格ライン: 12か月の通算回収1.0超かつ勝ち月8以上。\n\n"
    if B.empty: md+='まだ該当なし（成績・オッズ・直前情報がそろった日から数える）。\n'
    else:
        t3=B[B.rule=='ob-cheap-v1']; t2=B[B.rule=='ob-cheap2-v1']
        md+='## ob-cheap-v1 A（3連単・①頭70%以上）\n'+table(t3)+'\n## ob-cheap-v1 B（3連単・①頭80%以上）\n'+table(t3[t3.p1>=.8])+'\n## ob-cheap2-v1（2連単・①頭60%以上）\n'+table(t2)+'\n## 明細（新しい順、直近200点）\n| 候補 | 日付 | 場R | 買い目 | 締切時オッズ | ①頭 | 外した艇 | 結果 | 払戻 |\n|---|---|---|---|---|---|---|---|---|\n'
        for r in B.sort_values(['hd','jcd','rno'],ascending=False).head(200).itertuples():
            md+=f"| {r.rule} | {r.hd} | {VENUES[r.jcd]}{r.rno}R | {r.cb} | {r.odds} | {r.p1:.0%} | {r.X}（売れすぎ{r.gap:+.2f}・モーター差{r.modiff:+.0f}・展示{r.xer:.0f}位） | {r.res or '未'} | {'' if pd.isna(r.pay) else int(r.pay)} |\n"
    Path(A.out).write_text(md,encoding='utf-8'); n3=0 if B.empty else B[B.rule=='ob-cheap-v1'].groupby(KEY).ngroups; n2=0 if B.empty else B[B.rule=='ob-cheap2-v1'].groupby(KEY).ngroups
    print(f'{A.out} を書いた。3連単 {n3}レース、2連単 {n2}レース')
