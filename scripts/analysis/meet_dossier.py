"""1つの節（例: 住之江 高松宮記念）の選手・モーター資料を作る。運用者の観戦・趣味用の資料。
使い方: python scripts/analysis/meet_dossier.py --jcd 12 --from 20261105 --to 20261110 --out notes/reads/suminoe-takamatsunomiya.md
"""
import argparse, json, sqlite3, sys
import pandas as pd
sys.path.insert(0, "scripts")
from common import VENUES

ap = argparse.ArgumentParser()
ap.add_argument("--jcd", required=True); ap.add_argument("--from", dest="d0", required=True); ap.add_argument("--to", dest="d1", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
a = ap.parse_args()
con = sqlite3.connect("data/boatrace.db", timeout=600)
e = pd.read_sql("select hd,rno,lane,racer_no,racer_name,grade,motor_no,boat_no,close_time,race_name,data from entries where jcd=? and hd between ? and ?",
                con, params=(a.jcd, a.d0, a.d1))
if e.empty:
    sys.exit("番組表がありません")
e["d"] = e.data.map(json.loads)
for k in ["branch", "age", "national_win_rate", "local_win_rate", "motor_2ren_rate", "boat_2ren_rate"]:
    e[k] = e.d.map(lambda x: x.get(k))
r = pd.read_sql("select hd,rno,lane,position,data from results where jcd=? and hd between ? and ?", con, params=(a.jcd, a.d0, a.d1))
if not r.empty:
    r["d"] = r.data.map(json.loads); r["ex"] = r.d.map(lambda x: x.get("exhibition_time")); r["st"] = r.d.map(lambda x: x.get("st_timing"))
    r["course"] = r.d.map(lambda x: x.get("entrance_position"))
    r["pos"] = pd.to_numeric(r.position, errors="coerce")
    r["exr"] = r.groupby(["hd", "rno"]).ex.rank(method="min")
    m = e.merge(r[["hd", "rno", "lane", "pos", "ex", "exr", "st", "course"]], on=["hd", "rno", "lane"], how="left")
else:
    m = e.assign(pos=None, ex=None, exr=None, st=None, course=None)
# モーターの直近（この場の過去120日、この節より前）
first = e.hd.min()
past = pd.read_sql("""select e.hd,e.motor_no,r.position,r.data from entries e join results r on r.hd=e.hd and r.jcd=e.jcd and r.rno=e.rno and r.lane=e.lane
                      where e.jcd=? and e.hd < ? and e.hd >= ?""", con, params=(a.jcd, first, str(int(first) - 400)))
mot = {}
if not past.empty:
    past["pos"] = pd.to_numeric(past.position, errors="coerce")
    past["ex"] = past.data.map(lambda s: json.loads(s).get("exhibition_time"))
    g = past.groupby("motor_no")
    mot = pd.DataFrame({"走": g.size(), "平均着": g.pos.mean(), "3着内": g.pos.apply(lambda s: (s <= 3).mean())}).to_dict("index")
racers = e.sort_values("hd").groupby("racer_no").agg(name=("racer_name", "first"), grade=("grade", "first"), branch=("branch", "first"),
                                                     age=("age", "first"), nwr=("national_win_rate", "first"), lwr=("local_win_rate", "first"),
                                                     motor=("motor_no", "first"), m2=("motor_2ren_rate", "first"), boat=("boat_no", "first")).reset_index()
lines = [f"# {a.title or VENUES.get(a.jcd, a.jcd) + ' ' + a.d0 + '〜'}", "",
         "運用者の観戦・趣味用の資料。毎晩、その日の成績を足して作り直す（`scripts/analysis/meet_dossier.py`）。前検の気配は `notes/reads/2026-11-05-suminoe.md`。", "",
         "## 選手とモーター（勝率順）", "",
         "| 選手 | 級 | 支部 | 年 | 全国勝率 | 当地勝率 | モーター | 公式2連率 | このモーターの当地直近（走・平均着・3着内率） | 今節の成績 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for x in racers.sort_values("nwr", ascending=False).itertuples():
    mm = mot.get(x.motor, {})
    mtxt = f"{mm.get('走', 0)}走・{mm.get('平均着', float('nan')):.2f}・{mm.get('3着内', float('nan')):.0%}" if mm else "-"
    my = m[(m.racer_no == x.racer_no)].sort_values(["hd", "rno"])
    res = " ".join(f"{int(p)}" if pd.notna(p) else "・" for p in my.pos) if my.pos.notna().any() else "-"
    lines.append(f"| {x.name} | {x.grade} | {x.branch or ''} | {x.age or ''} | {x.nwr} | {x.lwr} | {x.motor} | {x.m2} | {mtxt} | {res} |")
if r is not None and not r.empty:
    lines += ["", "## 今節の展示・成績（日別）", ""]
    for (hd, rno), g in m[m.pos.notna()].groupby(["hd", "rno"]):
        g = g.sort_values("lane")
        lines.append(f"- {hd[4:6]}/{hd[6:]} {rno}R {g.race_name.iloc[0]}: " + " ".join(
            f"{int(l)}{n}(M{mo} 展{ex}/{int(er) if pd.notna(er) else '-'}位 進{c} ST{st} →{int(p)}着)" for l, n, mo, ex, er, c, st, p in
            zip(g.lane, g.racer_name, g.motor_no, g.ex, g.exr, g.course, g.st, g.pos)))
days = sorted(e.hd.unique())
lines += ["", f"番組表がある日: {', '.join(d[4:6] + '/' + d[6:] for d in days)}"]
open(a.out, "w").write("\n".join(lines) + "\n")
print(f"wrote {a.out}: racers {len(racers)}")
