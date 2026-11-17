"""締切が近いレースを見張り、合成10〜20倍で妙味のある形が組めるレースを探す（運用者の遊び用、舟券は買わない）。

材料（g1_prep の資料: コース別成績・モーター・節間）＋展示（展示順位・スタ展）から各艇の1着の目安を作り、
いまの3連単オッズから出した市場の見込みと比べる。頭は「目安/市場 ≥ 1.15」の艇（1号艇も含む）。
2・3着は目安の確率が高い順に足し、合成が lo〜hi に入る形を作る。3-2-1（全場で大口が入る）は使わない。
目安は検証していない。オッズは締切前のもので動く。

  python scripts/analysis/scan_now.py --date 20261118 \
      --dossier "びわこ=11=notes/reads/biwako-taisho-2026-11.md" \
      --dossier "丸亀=15=notes/reads/marugame-kyogokusho-2026-11.md" \
      --budget 1000 --loop --seen /tmp/seen.txt
  --loop: 見つかるまで3分おきに見て、見つかったら出力して終わる（締切まで8〜22分のレースだけ見る）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import math
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scrape_web as sw  # noqa: E402
from pick_races import boat, load  # noqa: E402
from race_now import allocate  # noqa: E402
from value_scan import harville, win_scores  # noqa: E402

JST = dt.timezone(dt.timedelta(hours=9))
SKIP = {"3-2-1"}


def adjust(p, bi):
    ex = sorted(bi, key=lambda r: r["exhibition_time"])
    er = {r["lane"]: i + 1 for i, r in enumerate(ex)}
    st = sorted(bi, key=lambda r: (r.get("start_st") if r.get("start_st") is not None else 0.3))
    sr = {r["lane"]: i + 1 for i, r in enumerate(st)}
    w = [p[i] * math.exp(0.25 * (3.5 - er[i + 1]) / 2.5 + 0.15 * (3.5 - sr[i + 1]) / 2.5) for i in range(6)]
    t = sum(w)
    return [x / t for x in w], er, sr


def build(p, odds, head, lo, hi, maxpts):
    cands = []
    for b, c in itertools.permutations([i for i in range(6) if i != head], 2):
        k = f"{head+1}-{b+1}-{c+1}"
        if k in odds and k not in SKIP:
            cands.append((harville(p, head, b, c), k, odds[k]))
    cands.sort(reverse=True)
    best = None
    picked = []
    for q, k, o in cands:
        trial = picked + [(q, k, o)]
        syn = 1 / sum(1 / x[2] for x in trial)
        if syn < lo:
            continue
        picked = trial
        if syn <= hi and (best is None or sum(x[0] for x in picked) > best[1]):
            best = (list(picked), sum(x[0] for x in picked), syn)
        if len(picked) >= maxpts:
            break
    return best


def scan(a, S, seen):
    now = dt.datetime.now(JST)
    out = []
    for spec in a.dossier:
        vname, jcd, path = spec.split("=", 2)
        d = load(path)
        if d["pday"] != a.date[4:]:
            continue
        for r in d["prog"]:
            rno = int(r["R"])
            key = f"{jcd}-{rno}"
            if key in seen:
                continue
            hh, mm = map(int, r["締切"].split(":"))
            close = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            mins = (close - now).total_seconds() / 60
            if not (a.min_before <= mins <= a.max_before):
                continue
            bi = sw.parse_beforeinfo(sw.fetch("beforeinfo", a.date, jcd, rno, S) or "")
            if not bi or not all(x.get("exhibition_time") for x in bi):
                continue
            odds = {c: v for _, c, v, _ in sw.parse_odds3t(sw.fetch("odds3t", a.date, jcd, rno, S) or "") if v}
            if len(odds) < 100:
                continue
            seen.add(key)
            bs = [boat(d, i, r[str(i)]) for i in range(1, 7)]
            p, er, sr = adjust(win_scores(bs), bi)
            inv = {h: sum(1 / v for k, v in odds.items() if k[0] == str(h + 1) and k not in SKIP) for h in range(6)}
            tot = sum(inv.values())
            m = [inv[h] / tot for h in range(6)]
            best = None
            for h in range(6):
                ratio = p[h] / max(m[h], 1e-6)
                if ratio < a.min_ratio or p[h] < 0.12:
                    continue
                res = build(p, odds, h, a.lo, a.hi, a.maxpts)
                if res and (best is None or res[1] * res[2] > best[2][1] * best[2][2]):
                    best = (h, ratio, res)
            if best:
                h, ratio, (pts, hit, syn) = best
                syn2, rows = allocate(odds, [k for _, k, _ in pts], a.budget)
                line = (f"{vname}{rno}R 締切{r['締切']}（あと{mins:.0f}分） 頭{h+1}{bs[h]['name']} 目安{p[h]:.0%}/市場{m[h]:.0%}（比{ratio:.2f}） "
                        f"展示順位{er[h+1]} スタ展順位{sr[h+1]} 合成{syn2:.1f}倍 {len(rows)}点 目安の的中{hit:.0%}: "
                        + " ".join(f"{c} {s}円({v})" for c, v, s in rows))
                out.append(line)
            else:
                out.append(f"{vname}{rno}R 締切{r['締切']}: 条件に合う形なし（見送り候補）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--dossier", action="append", required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--lo", type=float, default=10)
    ap.add_argument("--hi", type=float, default=20)
    ap.add_argument("--maxpts", type=int, default=10)
    ap.add_argument("--min-ratio", type=float, default=1.15)
    ap.add_argument("--min-before", type=float, default=8)
    ap.add_argument("--max-before", type=float, default=22)
    ap.add_argument("--seen", default="")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--until", default="20:50")
    a = ap.parse_args()
    seen = set(Path(a.seen).read_text().split()) if a.seen and Path(a.seen).exists() else set()
    S = requests.Session()
    hh, mm = map(int, a.until.split(":"))
    while True:
        lines = scan(a, S, seen)
        if a.seen:
            Path(a.seen).write_text("\n".join(sorted(seen)))
        hits = [x for x in lines if "条件に合う形なし" not in x]
        for x in lines:
            print(x, flush=True)
        if not a.loop or hits:
            return 0
        now = dt.datetime.now(JST)
        if (now.hour, now.minute) >= (hh, mm):
            print("終了時刻", flush=True)
            return 0
        time.sleep(180)


if __name__ == "__main__":
    raise SystemExit(main())
