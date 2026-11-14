"""今日の穴狙い候補（紙上の ana-s5-v1・ana-s6-v1 の条件）を出す。お金は使わない前提の「判断材料」。

    python scripts/analysis/ana_today.py [YYYYMMDD]

番組表（公式 B ファイル）で前夜に分かる条件を絞り、まだ締切前のレースは boatrace.jp から展示と今のオッズを取って判定する。
今のオッズは確定前（締切までに動く）。紙上の採点は締切時オッズで行うので、ここでの判定とずれることがある。
"""
import subprocess, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import VENUES  # noqa: E402
from ingest import parse_schedule_text  # noqa: E402
import scrape_web as sw  # noqa: E402

JST = dt.timezone(dt.timedelta(hours=9)); now = dt.datetime.now(JST)
hd = sys.argv[1] if len(sys.argv) > 1 else now.strftime("%Y%m%d")
cache = ROOT / "data/lzh_cache/B"; cache.mkdir(parents=True, exist_ok=True)
f = cache / f"b{hd[2:]}.lzh"
if not f.exists():
    r = requests.get(f"https://www1.mbrace.or.jp/od2/B/{hd[:6]}/b{hd[2:]}.lzh", timeout=30); r.raise_for_status(); f.write_bytes(r.content)
text = subprocess.run(["7z", "x", "-so", str(f)], capture_output=True).stdout.decode("cp932", errors="replace")
e = pd.DataFrame(parse_schedule_text(text))
w = e.pivot_table(index=["jcd", "rno", "close_time"], columns="lane", values="motor_2ren_rate").add_prefix("mo").reset_index()
g = e.pivot_table(index=["jcd", "rno"], columns="lane", values="grade", aggfunc="first").add_prefix("gr").reset_index()
w = w.merge(g, on=["jcd", "rno"])
w["s5"] = w.mo1 <= 25
w["s6"] = w.gr1.isin(["B1", "B2"]) & (w.gr6 == "A1")
cand = w[w.s5 | w.s6].sort_values("close_time")
print(f"{hd} 全{len(w)}レース。前夜の条件に合う: S5（①モーター25%以下）{int(w.s5.sum())}、S6（①B級・⑥A1）{int(w.s6.sum())}。今 {now:%H:%M}")
ses = requests.Session(); out = []
for r in cand.itertuples():
    name = f"{VENUES[r.jcd]}{r.rno}R 締切{r.close_time}"
    if r.close_time <= now.strftime("%H:%M"):
        out.append((name, "締切済み", "")); continue
    bi = sw.parse_beforeinfo(sw.fetch("beforeinfo", hd, r.jcd, r.rno, ses) or "")
    od = sw.parse_odds3t(sw.fetch("odds3t", hd, r.jcd, r.rno, ses) or "")
    et = {b["lane"]: b.get("exhibition_time") for b in bi if b.get("exhibition_time")}
    q = {cb.replace("-", ""): 1 / o for _, cb, o, _ in od if o}
    if len(et) < 6 or len(q) < 100:
        why = []
        if r.s5: why.append(f"S5待ち（①モーター{r.mo1:.0f}%、⑤モーター{r.mo5:.0f}%）")
        if r.s6: why.append("S6待ち（①B級・⑥A1）")
        out.append((name, "展示・オッズ待ち", "、".join(why))); continue
    rank = pd.Series(et).rank(method="min"); S = sum(q.values()); p1 = sum(v for k, v in q.items() if k[0] == "1") / S
    msg = []
    if r.s5:
        ok = (p1 >= .5) and (rank[5] <= 2 or r.mo5 >= 40); tk = ["512", "513", "514"]; co = 1 / sum(q[t] for t in tk)
        msg.append(("◎S5 5-1-2・5-1-3・5-1-4" if ok else "×S5") + f"（①頭シェア{p1:.0%}、①モーター{r.mo1:.0f}%、⑤展示{rank[5]:.0f}位、⑤モーター{r.mo5:.0f}%、合成{co:.0f}倍）")
    if r.s6:
        tk = ["321", "324", "325", "326"]; co = 1 / sum(q[t] for t in tk); ok = (rank[3] <= 3) and (15 <= co <= 60)
        msg.append(("◎S6 3-2-全" if ok else "×S6") + f"（③展示{rank[3]:.0f}位、合成{co:.0f}倍）")
    out.append((name, "判定", "、".join(msg)))
for a, b, c in out:
    if b != "締切済み": print(f"{a} [{b}] {c}")
print(f"締切済み {sum(1 for x in out if x[1]=='締切済み')}レース")
