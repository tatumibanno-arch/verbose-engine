"""選手の「そのコースでの勝ち方・負け方」（決まり手）を集計する。

    python scripts/analysis/kimarite_course.py 20261118 "11:4,5,9,11" "15:6,8,9,11" [--days 365]

- 期間: 対象日の前日までの直近 days 日（既定365日、全場）。レース後に分かる情報だけを過去分として使う。
- コース: results.data の entrance_position（実際の進入）。対象日の艇は枠＝コースとして当てる。
- 決まり手: data/parquet/kimarite_*.parquet（勝者の決まり手。レースに1つ）。
- 1コース: 逃げ率／差された率（勝者が差し・まくり差し）／まくられた率（勝者がまくり）
- 2コース: 差し率／まくり率（自分が勝った決まり手）／1号艇を逃がした率（そのレースの決まり手が逃げ）
- 3〜6コース: まくり率／まくり差し率／差し率（自分が勝った決まり手）
- 率はすべて「そのコースでの出走数」が分母。20走未満は注記。
出力は Markdown（標準出力）。
"""
import argparse, json, sqlite3, sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN_N = 20
COLS = {1: [("逃げ", "w_ng"), ("差され", "l_ss"), ("まくられ", "l_mk")],
        2: [("差し", "w_ss"), ("まくり", "w_mk"), ("1を逃がし", "nige")]}
for c in range(3, 7):
    COLS[c] = [("まくり", "w_mk"), ("まくり差し", "w_mz"), ("差し", "w_ss")]


def load(hd0, hd1):
    con = sqlite3.connect(ROOT / "data/boatrace.db")
    r = pd.read_sql("select hd, jcd, rno, lane, position, data from results where hd between ? and ?", con, params=(hd0, hd1))
    d = r.data.map(json.loads)
    r["racer"] = d.map(lambda x: x.get("racer_number"))
    r["course"] = d.map(lambda x: x.get("entrance_position"))
    r = r.dropna(subset=["racer", "course"]).astype({"racer": int, "course": int})
    r["jcd"] = r.jcd.astype(int)
    k = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "data/parquet").glob("kimarite_*.parquet"))])
    k = k[(k.hd >= hd0) & (k.hd <= hd1)][["hd", "jcd", "rno", "kimarite"]]
    m = r.merge(k, on=["hd", "jcd", "rno"], how="inner")
    win = m.position == 1; km = m.kimarite
    m["win"] = win; m["top3"] = m.position.fillna(9) <= 3
    m["w_ng"] = win & (km == "逃げ"); m["w_ss"] = win & (km == "差し")
    m["w_mk"] = win & (km == "まくり"); m["w_mz"] = win & (km == "まくり差し")
    m["l_ss"] = ~win & km.isin(["差し", "まくり差し"]); m["l_mk"] = ~win & (km == "まくり")
    m["nige"] = km == "逃げ"
    return m, con


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("hd"); ap.add_argument("races", nargs="+"); ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args()
    d = datetime.strptime(a.hd, "%Y%m%d")
    hd0 = (d - timedelta(days=a.days)).strftime("%Y%m%d"); hd1 = (d - timedelta(days=1)).strftime("%Y%m%d")
    m, con = load(hd0, hd1)
    flags = ["win", "top3", "w_ng", "w_ss", "w_mk", "w_mz", "l_ss", "l_mk", "nige"]
    agg = m.groupby(["racer", "course"])[flags].agg(["sum"]).droplevel(1, axis=1)
    agg["n"] = m.groupby(["racer", "course"]).size()
    base = m.groupby("course")[flags].mean()
    print(f"期間 {hd0}〜{hd1}（全場、決まり手つき {m[['hd','jcd','rno']].drop_duplicates().shape[0]} レース）\n")
    print("全選手のコース別平均（%）: " + " / ".join(
        f"{c}コース 1着{base.loc[c,'win']*100:.0f} " + " ".join(f"{lab}{base.loc[c,col]*100:.0f}" for lab, col in COLS[c]) for c in range(1, 7)) + "\n")
    for spec in a.races:
        jcd, rnos = spec.split(":")
        for rno in map(int, rnos.split(",")):
            e = pd.read_sql("select lane, racer_no, racer_name, grade from entries where hd=? and cast(jcd as int)=? and rno=? order by lane",
                            con, params=(a.hd, int(jcd), rno))
            print(f"### {int(jcd):02d} {rno}R\n")
            print("| 枠 | 選手 | 級 | 走 | 1着% | 3内% | 決まり手（%） |\n|---|---|---|---|---|---|---|")
            for _, x in e.iterrows():
                c = int(x.lane); key = (int(x.racer_no), c)
                if key in agg.index:
                    s = agg.loc[key]; n = int(s.n)
                    kim = " ".join(f"{lab}{s[col]/n*100:.0f}" for lab, col in COLS[c])
                    row = f"{n}{'※' if n < MIN_N else ''} | {s.win/n*100:.0f} | {s.top3/n*100:.0f} | {kim}"
                else:
                    row = "0※ | - | - | -"
                print(f"| {c} | {x.racer_name.replace(' ', '').replace('　', '')} | {x.grade} | {row} |")
            print()


if __name__ == "__main__":
    sys.exit(main())
