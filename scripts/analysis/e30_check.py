"""新燃料E30（バイオエタノール30%）の導入前後で、1号艇の1着率・万舟・払戻・スタートが変わったか。
導入日: 公式発表は2026-03-30。各場のモーター入替に合わせて切り替え。入替日はデータ（モーター2連率が一斉に0になる日）から検出し、
note記事（2026-06-23、パック氏）に載っている8場の日付と一致を確認。記事に無い場は「推定」「不明」と区別する。
比べ方: 導入後〜2026-11-23 と、その1年前の同じ期間（同じ場）。季節の影響を消すため。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts'); from common import VENUES
c=sqlite3.connect('data/boatrace.db'); END='20261123'
CONF={'06':'20260409','21':'20260416','13':'20260417','05':'20260418','18':'20260420','19':'20260429','03':'20260511','04':'20260618'}   # 記事に日付あり
EST={'02':'20260806','07':'20260811','23':'20260916','15':'20260917','08':'20261006','17':'20261020'}                                  # 記事後のモーター入替（推定）
EARLY={'11':'20250501','24':'20250622'}                                                                                                   # 先行テスト場（2025年5月以降。入替日から数える）
NOT=['01','09','16','20','10','12','22']                                                                                                 # 2026-04-09以降に入替なし＝未導入とみなす
UNK={'14':'20260411','24b':None}
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20230101'",c).drop_duplicates(['hd','jcd','rno'])
r=pd.read_sql("select hd,jcd,rno,lane,position,json_extract(data,'$.st_timing') st from results where hd>='20230101'",c); r['st']=pd.to_numeric(r.st,errors='coerce'); r['out']=r.position.isna()
rs=r.groupby(['hd','jcd','rno']).agg(st=('st','mean'),out=('out','sum')).reset_index(); p=p.merge(rs,on=['hd','jcd','rno'],how='left')
p['w1']=p.res.str[0]=='1'; p['c3']=p.res.str[0].isin(['3','4']); p['man']=p.payout>=10000
def stat(s): return dict(n=len(s),w1=s.w1.mean(),c34=s.c3.mean(),man=s.man.mean(),med=s.payout.median(),st=s.st.mean(),out=s.out.mean())
def shift(d,y): return str(int(d[:4])-y)+d[4:]
def row(j,d0,d1=END):
    a=stat(p[(p.jcd==j)&(p.hd>=d0)&(p.hd<=d1)]); b=stat(p[(p.jcd==j)&(p.hd>=shift(d0,1))&(p.hd<=shift(d1,1))]); return a,b
def table(nm,M):
    print('\n==',nm); tot=[]
    for j,d in M.items():
        a,b=row(j,d); tot.append((a,b))
        print(f"{VENUES[j]}（{d[4:6]}/{d[6:]}〜）レース {a['n']}/{b['n']} | 1号艇1着 {b['w1']:.3f}→{a['w1']:.3f}（{a['w1']-b['w1']:+.3f}） ③④頭 {b['c34']:.3f}→{a['c34']:.3f} 万舟 {b['man']:.3f}→{a['man']:.3f}（{a['man']-b['man']:+.3f}） 中央値 {b['med']:.0f}→{a['med']:.0f} 平均ST {b['st']:.3f}→{a['st']:.3f} 失格等/レース {b['out']:.3f}→{a['out']:.3f}")
    w=lambda k,i: sum(t[i][k]*t[i]['n'] for t in tot)/sum(t[i]['n'] for t in tot)
    print(f"  合計 レース {sum(t[0]['n'] for t in tot)} | 1号艇1着 {w('w1',1):.3f}→{w('w1',0):.3f}（{w('w1',0)-w('w1',1):+.3f}） ③④頭 {w('c34',1):.3f}→{w('c34',0):.3f} 万舟 {w('man',1):.3f}→{w('man',0):.3f}（{w('man',0)-w('man',1):+.3f}） 平均ST {w('st',1):.3f}→{w('st',0):.3f} 失格等 {w('out',1):.3f}→{w('out',0):.3f}")
table('導入済み（記事に日付あり）: 前年同期 → 導入後',CONF); table('導入済みと推定（記事の後にモーター入替）',EST)
table('未導入とみなす場（比較用。2026-04-20〜を前年同期と比べる）',{j:'20260420' for j in NOT})
print('\n== 先行テスト場（びわこ・大村）: 入替の前年同期 → 入替後1年')
for j,d in EARLY.items():
    a=stat(p[(p.jcd==j)&(p.hd>=d)&(p.hd<shift(d,-1))]); b=stat(p[(p.jcd==j)&(p.hd>=shift(d,1))&(p.hd<d)])
    print(f"{VENUES[j]}（{d}〜）レース {a['n']}/{b['n']} | 1号艇1着 {b['w1']:.3f}→{a['w1']:.3f} ③④頭 {b['c34']:.3f}→{a['c34']:.3f} 万舟 {b['man']:.3f}→{a['man']:.3f} 中央値 {b['med']:.0f}→{a['med']:.0f} 平均ST {b['st']:.3f}→{a['st']:.3f} 失格等 {b['out']:.3f}→{a['out']:.3f}")
print('\n== 導入直後の30日だけ（記事に日付ありの8場）')
tot=[]
for j,d in CONF.items():
    d1=(pd.Timestamp(d)+pd.Timedelta(days=30)).strftime('%Y%m%d'); a,b=row(j,d,d1); tot.append((a,b))
w=lambda k,i: sum(t[i][k]*t[i]['n'] for t in tot)/sum(t[i]['n'] for t in tot)
print(f"レース {sum(t[0]['n'] for t in tot)} | 1号艇1着 {w('w1',1):.3f}→{w('w1',0):.3f} ③④頭 {w('c34',1):.3f}→{w('c34',0):.3f} 万舟 {w('man',1):.3f}→{w('man',0):.3f} 平均ST {w('st',1):.3f}→{w('st',0):.3f} 失格等 {w('out',1):.3f}→{w('out',0):.3f}")
