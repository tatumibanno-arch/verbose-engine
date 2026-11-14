"""3連単の「市場の偏り」探し（2026-11-14）。締切時オッズで、どんな目を100円ずつ買うと回収が高いかを区分ごとに出す。
発見用＝2025-03〜2026-04、検証用＝2026-05〜。締切時オッズ＝確定に近いので、払戻の計算は実際とほぼ同じ（買う時点のオッズではない点に注意）。
区分: ①オッズ帯 ②頭の艇×オッズ帯 ③場×人気帯 ④レース番号 ⑤展示1位が頭か×オッズ帯 ⑥進入が枠なりか×頭①×オッズ帯 ⑦市場の①頭シェア×頭①×オッズ帯"""
import sqlite3, numpy as np, pandas as pd
SPLIT="20260501"
c=sqlite3.connect("data/boatrace.db")
o=pd.read_sql("select hd,jcd,rno,combination as cb,odds from odds where ticket_type='sanrensho' and odds>0",c)
o["jcd"]=o.jcd.astype(int); o["hd"]=o.hd.astype(str)
p=pd.read_sql("select hd,jcd,rno,combination as cb,payout from payouts where ticket_type='sanrensho' and hd>='20250301'",c)
p["jcd"]=p.jcd.astype(int); p["hd"]=p.hd.astype(str)
o=o.merge(p[["hd","jcd","rno"]].drop_duplicates(),on=["hd","jcd","rno"])  # 結果のあるレースだけ
o=o.merge(p,on=["hd","jcd","rno","cb"],how="left"); o["ret"]=o.payout.fillna(0)/100
o["h"]=o.cb.str[0].astype(int); o["s"]=o.cb.str[2].astype(int); o["per"]=np.where(o.hd<SPLIT,"発見","検証")
o["band"]=pd.cut(o.odds,[0,3,5,8,12,20,35,60,100,300,1e9],labels=["〜3","3〜5","5〜8","8〜12","12〜20","20〜35","35〜60","60〜100","100〜300","300〜"])
tot=o.assign(q=1/o.odds).groupby(["hd","jcd","rno"]).q.transform("sum"); o["mp"]=(1/o.odds)/tot
m1=o[o.h==1].groupby(["hd","jcd","rno"]).mp.sum().rename("m1").reset_index(); o=o.merge(m1,on=["hd","jcd","rno"],how="left")
print("レース数",o.groupby(["hd","jcd","rno"]).ngroups,"行",len(o))
def tab(d,keys,minn=3000):
    g=d.groupby(keys+["per"],observed=True).agg(n=("ret","size"),roi=("ret","mean")).reset_index()
    w=g.pivot_table(index=keys,columns="per",values=["n","roi"],observed=True)
    w=w[(w[("n","発見")]>=minn)]; w[("roi","発見")]=w[("roi","発見")].round(3); w[("roi","検証")]=w[("roi","検証")].round(3)
    return w
pd.set_option("display.width",200); pd.set_option("display.max_rows",400)
print("== ① オッズ帯 =="); print(tab(o,["band"]).to_string())
print("== ② 頭の艇×オッズ帯 =="); print(tab(o,["h","band"]).to_string())
print("== ②' 頭①・2着の艇×オッズ帯（〜35倍） =="); print(tab(o[(o.h==1)&(o.odds<35)],["s","band"]).to_string())
print("== ③ 場×人気帯（〜12倍） =="); print(tab(o[o.odds<12],["jcd"]).to_string())
print("== ④ レース番号×（〜12倍） =="); print(tab(o[o.odds<12],["rno"]).to_string())
bi=pd.read_sql("select hd,jcd,rno,lane,exhibition_time as et,start_course as sc from beforeinfo",c); bi["jcd"]=bi.jcd.astype(int); bi["hd"]=bi.hd.astype(str)
bi["erank"]=bi.groupby(["hd","jcd","rno"]).et.rank(method="min")
wk=bi.dropna(subset=["sc"]).assign(ok=lambda d:d.sc==d.lane).groupby(["hd","jcd","rno"]).ok.agg(["all","size"]).reset_index(); wk["waku"]=wk["all"]&(wk["size"]==6)
x=o.merge(bi[["hd","jcd","rno","lane","erank"]].rename(columns={"lane":"h","erank":"h_erank"}),on=["hd","jcd","rno","h"],how="inner").merge(wk[["hd","jcd","rno","waku"]],on=["hd","jcd","rno"],how="left")
x["he"]=pd.cut(x.h_erank,[0,1,2,4,6],labels=["展示1位","2位","3〜4位","5〜6位"])
print("== ⑤ 頭の艇の展示順位×オッズ帯 =="); print(tab(x,["he","band"]).to_string())
print("== ⑤' 頭①の展示順位×オッズ帯 =="); print(tab(x[x.h==1],["he","band"],1500).to_string())
print("== ⑥ 進入枠なりか×頭①か×オッズ帯（〜35倍） =="); x["h1"]=x.h==1; print(tab(x[x.odds<35].dropna(subset=["waku"]),["waku","h1","band"],1500).to_string())
o["m1b"]=pd.cut(o.m1,[0,.3,.45,.6,.75,1])
print("== ⑦ 市場の①頭シェア×頭①か×オッズ帯（〜35倍） =="); o["h1"]=o.h==1; print(tab(o[o.odds<35],["m1b","h1","band"],2000).to_string())
