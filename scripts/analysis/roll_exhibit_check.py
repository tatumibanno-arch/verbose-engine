"""転がし v2 の展示ルールの検証（過去データ）。
モデル予測（ウォークフォワードの pred_YYYY-MM.npz）＋ beforeinfo の展示タイム＋ results の着順で、
「モデルの①1着確率>=0.6」のレースについて、展示条件ごとに ①1着率・モデル上位の目（確率合計>=50%まで）の的中率を比べる。
展示は締切前に分かる情報なので、締切前アラートと同じ扱い（当日運用でのみ使う）。"""
import sys, glob, itertools, sqlite3, numpy as np, pandas as pd
W="/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/pred"
PERMS=list(itertools.permutations(range(1,7),3))
IDX={c:k for k,c in enumerate(PERMS)}
parts=[]
for f in sorted(glob.glob(f"{W}/pred_202[56]-*.npz")):
    z=np.load(f); P=z["P"]
    p1=P[:,[k for k,c in enumerate(PERMS) if c[0]==1]].sum(1)
    parts.append(pd.DataFrame({"hd":z["hd"].astype(str),"jcd":pd.Series(z["jcd"].astype(str)).str.zfill(2),"rno":z["rno"].astype(int),"p1":p1,"i":np.arange(len(P)),"f":f}))
pr=pd.concat(parts,ignore_index=True)
PS={f:np.load(f)["P"] for f in pr.f.unique()}
c=sqlite3.connect("data/boatrace.db")
bi=pd.read_sql("select hd,jcd,rno,lane,exhibition_time ex from beforeinfo where exhibition_time is not null",c)
rs=pd.read_sql("select hd,jcd,rno,lane,position from results where position between 1 and 3",c)
for d in (bi,rs): d["jcd"]=d.jcd.astype(str).str.zfill(2); d["hd"]=d.hd.astype(str)
piv=bi.pivot_table(index=["hd","jcd","rno"],columns="lane",values="ex")
piv=piv.dropna()
rk=piv.rank(axis=1,method="min")
feat=pd.DataFrame({"r1":rk[1],"ex1":piv[1],"outfast":(piv[[4,5,6]].min(axis=1)<=piv[1]-0.05)&(piv[[4,5,6]].min(axis=1)==piv.min(axis=1))}).reset_index()
top3=rs.sort_values("position").groupby(["hd","jcd","rno"]).lane.apply(tuple).reset_index(name="top3")
df=pr.merge(feat,on=["hd","jcd","rno"]).merge(top3,on=["hd","jcd","rno"])
df=df[df.top3.apply(len)==3]
df["win1"]=df.top3.apply(lambda t:t[0]==1)
def _hit(r):
    P=PS[r.f][r.i]; pa=P[IDX[r.top3]]
    return P[P>pa].sum()<0.5
df["hit"]=df.apply(_hit,axis=1)
base=df[df.p1>=0.6]
def show(name,d): print(f"{name:34s} {len(d):6d}R ①1着{d.win1.mean()*100:5.1f}% モデル上位50%の的中{d.hit.mean()*100:5.1f}%")
print("期間",df.hd.min(),"〜",df.hd.max())
show("①1着確率>=0.6 全部",base)
show("  うち①展示5〜6位",base[base.r1>=5])
show("  うち①展示1〜4位",base[base.r1<=4])
show("  うち外(4〜6)が展示1位で①より.05速い",base[base.outfast])
show("  v2で残る（両方に当たらない）",base[(base.r1<=4)&(~base.outfast)])
show("  v2で外す（どちらかに当たる）",base[(base.r1>=5)|(base.outfast)])
