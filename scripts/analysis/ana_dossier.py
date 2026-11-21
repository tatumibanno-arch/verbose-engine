"""今日のレースの材料を1レースずつまとめる（穴狙いの展開読み用）。前夜に分かる情報＋今節の成績。

    python scripts/analysis/ana_dossier.py OUTDIR [YYYYMMDD]     → OUTDIR/dossier_{jcd}.md（場ごと）と OUTDIR/index.txt
"""
import subprocess, sys, re, json, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, requests, duckdb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import VENUES  # noqa: E402
from ingest import parse_schedule_text  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
JST = dt.timezone(dt.timedelta(hours=9)); now = dt.datetime.now(JST)
hd = sys.argv[2] if len(sys.argv) > 2 else now.strftime("%Y%m%d")
P = str(ROOT / "data/parquet"); c = duckdb.connect()


def lzh(kind, d):
    cache = ROOT / f"data/lzh_cache/{kind}"; cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{kind.lower()}{d[2:]}.lzh"
    if not f.exists():
        # Python の requests は公式サイトに 403 で弾かれることがある（2026-11-21）。curl で1本ずつ取る。
        url = f"https://www1.mbrace.or.jp/od2/{kind}/{d[:6]}/{kind.lower()}{d[2:]}.lzh"
        r = subprocess.run(["curl", "-sS", "-f", "-m", "90", "--retry", "2", "-o", str(f), url], capture_output=True)
        if r.returncode != 0:
            f.unlink(missing_ok=True); return ""
    return subprocess.run(["7z", "x", "-so", str(f)], capture_output=True).stdout.decode("cp932", errors="replace")


e = pd.DataFrame(parse_schedule_text(lzh("B", hd)))
# ---- 今節のST・着順（直近7日の成績ファイルから、同じ場・同じ選手）----
rows = []
ROW = re.compile(r"^\s{2}(\S{2})\s+([1-6])\s(\d{4})\s.*?(\d\.\d\d)\s+([1-6])\s+([FL]?\s?\d?\.\d\d)")
for i in range(1, 8):
    d = (dt.datetime.strptime(hd, "%Y%m%d") - dt.timedelta(days=i)).strftime("%Y%m%d"); jcd = None
    for line in lzh("K", d).splitlines():
        m = re.match(r"^(\d{2})KBGN", line)
        if m: jcd = m.group(1); continue
        m = ROW.match(line)
        if m and jcd:
            st = m.group(6).replace(" ", "")
            rows.append((d, jcd, int(m.group(3)), m.group(1), int(m.group(5)), -float(st[1:]) if st[0] == "F" else (np.nan if st[0] == "L" else float(st))))
k = pd.DataFrame(rows, columns=["d", "jcd", "racer_no", "pos", "course", "st"])
meet = k.groupby(["jcd", "racer_no"]).agg(今節走=("st", "size"), 今節ST=("st", "mean"), 今節着=("pos", lambda s: "".join(x.lstrip("0") if x.isdigit() else x[0] for x in s))).reset_index()
e = e.merge(meet, on=["jcd", "racer_no"], how="left")
# ---- 期別のコース別成績 ----
st = c.sql(f"select cast(racer_no as int) racer_no, data from '{P}/racer_stats_*.parquet' where hd=(select max(hd) from '{P}/racer_stats_*.parquet')").df()
J = {r: json.loads(d) for r, d in st.itertuples(index=False)}
e["枠1着率"] = [J.get(r, {}).get(f"c{l}_win_rate") for r, l in zip(e.racer_no, e.lane)]
e["枠平均ST"] = [J.get(r, {}).get(f"c{l}_avg_st") for r, l in zip(e.racer_no, e.lane)]
e["枠走数"] = [J.get(r, {}).get(f"c{l}_starts") for r, l in zip(e.racer_no, e.lane)]
# ---- 決まり手の癖（過去365日、その枠＝コース）----
h = c.sql(f"""select cast(json_extract(r.data,'$.racer_number') as int) racer_no, cast(json_extract(r.data,'$.entrance_position') as int) lane, count(*) n,
  sum(case when r.position=1 and k.kimarite='逃げ' then 1 else 0 end) ng, sum(case when r.position=1 and k.kimarite='まくり' then 1 else 0 end) mk,
  sum(case when r.position=1 and k.kimarite='まくり差し' then 1 else 0 end) mz, sum(case when r.position=1 and k.kimarite='差し' then 1 else 0 end) ss,
  sum(case when coalesce(r.position,9)<>1 and k.kimarite='まくり' then 1 else 0 end) lmk, sum(case when coalesce(r.position,9)<>1 and k.kimarite in ('差し','まくり差し') then 1 else 0 end) lss
  from '{P}/results_*.parquet' r join '{P}/kimarite_*.parquet' k on k.hd=r.hd and k.jcd=cast(r.jcd as int) and k.rno=r.rno
  where r.hd>=strftime(strptime('{hd}','%Y%m%d') - interval 365 day,'%Y%m%d') group by 1,2""").df()
e = e.merge(h, on=["racer_no", "lane"], how="left")


def habit(r):
    if not r.n or np.isnan(r.n): return "癖データなし"
    n = r.n; p = lambda x: f"{x/n*100:.0f}%"
    if r.lane == 1: return f"1コース{n:.0f}走: 逃げ{p(r.ng)} まくられ{p(r.lmk)} 差され{p(r.lss)}"
    return f"{r.lane}コース{n:.0f}走: まくり{p(r.mk)} まくり差し{p(r.mz)} 差し{p(r.ss)}"


idx = []
for jcd, g in e.groupby("jcd"):
    L = [f"# {VENUES[jcd]}（場コード{jcd}） {hd}", "平均の目安: 1コース 逃げ52%・まくられ16%・差され24%。まくり勝ち 2コース4%・3コース5%・4コース5%・5コース1%。まくり差し勝ち 3コース5%・4コース3%・5コース4%。差し勝ち 2コース9%。", ""]
    for (rno, ct, rn), r in g.groupby(["rno", "close_time", "race_name"]):
        L.append(f"## {rno}R {rn} 締切{ct}")
        for x in r.sort_values("lane").itertuples():
            ms = f"今節{x.今節走:.0f}走 ST平均{x.今節ST:.2f} 着{x.今節着}" if x.今節走 == x.今節走 else "今節成績なし(初日か取得なし)"
            cw = f"枠1着率{x.枠1着率:.0f}%(ST{x.枠平均ST:.2f},{x.枠走数:.0f}走)" if x.枠1着率 == x.枠1着率 and x.枠1着率 is not None else "枠成績なし"
            L.append(f"- {x.lane} {x.racer_name} {x.grade} 全国{x.national_win_rate} 当地{x.local_win_rate} モーター{x.motor_2ren_rate}% | {ms} | 期別:{cw} | 癖:{habit(x)}")
        L.append(""); idx.append((ct, jcd, VENUES[jcd], rno))
    (out / f"dossier_{jcd}.md").write_text("\n".join(L), encoding="utf-8")
idx.sort(); (out / "index.txt").write_text("\n".join(f"{a} {b} {c_}{d}R" for a, b, c_, d in idx), encoding="utf-8")
left = [x for x in idx if x[0] > now.strftime("%H:%M")]
print(f"{hd} {e.jcd.nunique()}場 {len(idx)}レース、締切前 {len(left)}。今 {now:%H:%M}")
print(pd.Series([f"{x[1]}{x[2]}" for x in left]).value_counts().sort_index().to_string())
print("今節成績あり", e.今節走.notna().mean().round(2), "癖あり", e.n.notna().mean().round(2), "期別あり", e.枠1着率.notna().mean().round(2))
