"""場ごとの「深い直前分析」の材料を前夜に作る（data/live/<date>_deep_<jcd>.json）。
当日の締切前に live_alert.py がこれに展示・風・オッズを足して LINE に送る。
使い方: python scripts/analysis/deep_prep.py --date 20261105 --jcd 12
"""
import argparse, json, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/analysis")
import trifecta as T

ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); ap.add_argument("--jcd", required=True)
ap.add_argument("--meet-from", help="今節の初日（今節の成績を出す）")
a = ap.parse_args(); D, J = a.date, a.jcd
con = sqlite3.connect("data/boatrace.db", timeout=600)
e = pd.read_sql("select rno,lane,racer_no,racer_name,grade,motor_no,close_time,race_name,data from entries where hd=? and jcd=? order by rno,lane", con, params=(D, J))
if e.empty: sys.exit("番組表なし")
e["d"] = e.data.map(json.loads)
# 期別成績（コース別）: D より前の最新
rs = pd.read_sql("select racer_no, hd, data from racer_stats where hd < ? order by hd", con, params=(D,)).drop_duplicates("racer_no", keep="last").set_index("racer_no")
# モーターの当地直近（今節より前の400日）
meet0 = a.meet_from or D
past = pd.read_sql("""select e.motor_no, r.position from entries e join results r on r.hd=e.hd and r.jcd=e.jcd and r.rno=e.rno and r.lane=e.lane
                      where e.jcd=? and e.hd < ? and e.hd >= ?""", con, params=(J, meet0, str(int(meet0) - 10000)))
past["pos"] = pd.to_numeric(past.position, errors="coerce")
mot = past.groupby("motor_no").pos.agg(lambda s: (s <= 3).mean()).to_dict()
# 今節の成績（前日まで）
cur = pd.read_sql("""select e.racer_no, r.hd, r.rno, r.position, r.data from entries e join results r on r.hd=e.hd and r.jcd=e.jcd and r.rno=e.rno and r.lane=e.lane
                     where e.jcd=? and e.hd >= ? and e.hd < ? order by r.hd, r.rno""", con, params=(J, meet0, D))
cur["ex"] = cur.data.map(lambda s: json.loads(s).get("exhibition_time"))
cur["exr"] = cur.groupby(["hd", "rno"]).ex.rank(method="min")
# 風（この場の過去10年、風速別の1号艇1着率）
w = pd.read_sql("select r.wind_speed ws, x.lane from races r join results x on x.hd=r.hd and x.jcd=r.jcd and x.rno=r.rno and x.position='1' where r.jcd=? and r.hd>='20150101'", con, params=(J,))
bins = [-1, 1, 2, 3, 4, 5, 99]; w["b"] = pd.cut(w.ws, bins, labels=False)
wind = {int(k): round(float(v), 3) for k, v in w.groupby("b").lane.apply(lambda s: (s == 1).mean()).items()}
# モデル（前夜の着順確率）
pred = Path(T.DEFAULT_WORK) / "pred" / f"today_{D}.npz"
z = np.load(pred) if pred.exists() else None
flags = yaml.safe_load(open(f"notes/reads/flags_{J}.yaml")) if Path(f"notes/reads/flags_{J}.yaml").exists() else {}
out = {"wind_bins": bins, "wind_l1": wind, "races": {}}
for rno, g in e.groupby("rno"):
    lanes = []
    for x in g.itertuples():
        st = json.loads(rs.data[x.racer_no]) if x.racer_no in rs.index else {}
        c = f"c{x.lane}_"
        my = cur[cur.racer_no == x.racer_no]
        nm = x.racer_name.replace(" ", "")
        mno = str(x.motor_no)
        good = [str(v) for v in flags.get("good_motors") or []]; bad = [str(v) for v in flags.get("bad_motors") or []]
        flag = ("◎" if nm in (flags.get("good_racers") or []) or mno in good else
                "△" if nm in (flags.get("bad_racers") or []) or mno in bad else "")
        lanes.append({"lane": x.lane, "name": x.racer_name, "grade": x.grade, "nwr": x.d.get("national_win_rate"),
                      "motor": int(x.motor_no), "m2": x.d.get("motor_2ren_rate"), "m_local3": round(mot.get(x.motor_no, np.nan), 2) if x.motor_no in mot else None,
                      "c_win": st.get(c + "win_rate"), "c_top2": st.get(c + "top2_rate"), "c_st": st.get(c + "avg_st"), "c_strank": st.get(c + "avg_start_rank"),
                      "meet": " ".join(str(int(p)) if str(p).isdigit() else "・" for p in my.position) or "-",
                      "meet_exr": round(float(my.exr.mean()), 1) if my.exr.notna().any() else None,
                      "flag": flag, "flag_note": {str(k): v for k, v in (flags.get("notes") or {}).items()}.get(mno, "")})
    item = {"close": g.close_time.iloc[0], "race_name": g.race_name.iloc[0], "lanes": lanes}
    if z is not None:
        k = [i for i in range(len(z["jcd"])) if str(z["jcd"][i]).zfill(2) == J and int(z["rno"][i]) == int(rno)]
        if k:
            P = z["P"][k[0]].astype(float); P1 = z["P1"][k[0]].astype(float)
            top = np.argsort(-P)[:10]
            item["p1"] = [round(float(v), 3) for v in P1]
            item["top"] = [["-".join(map(str, T.PERMS[t] + 1)), round(float(P[t]), 4)] for t in top]
    out["races"][str(int(rno))] = item
p = Path(f"data/live/{D}_deep_{J}.json"); p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"wrote {p}: {len(out['races'])} races")
