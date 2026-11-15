"""当日の G1・SG の全レースを、いまのオッズと前夜までの材料で見比べ、「1,000円で合成15〜20倍」が組めそうなレースを並べる。

運用者の遊び用。前夜までの材料（g1_prep.py の資料: 選手のコース別成績・モーター・節間）から各艇の1着の目安を作り、
いまの3連単オッズから出した「市場の1着の見込み」と比べて、材料のわりに売れていない頭を探す。
目安は検証していない（当たりを約束しない）。オッズは締切前のもの（確定オッズとは違う）で、展示の後に変わる。

  python scripts/analysis/value_scan.py --date 20261116 \
      --dossier "びわこ=11=notes/reads/biwako-taisho-2026-11.md" \
      --dossier "丸亀=15=notes/reads/marugame-kyogokusho-2026-11.md" \
      --out notes/reads/value-2026-11-16.md [--budget 1000 --lo 15 --hi 20]
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scrape_web as sw  # noqa: E402
from pick_races import boat, course_cell, load, num  # noqa: E402

VENUE_C1 = [55, 13.5, 13.5, 10, 6.5, 2]  # 場のコース別1着率（直近1年、両場ほぼ同じ）の目安


def win_scores(bs):
    s = []
    for b in bs:
        c = course_cell(b["ra"].get(f"{b['lane']}コース"))
        base = VENUE_C1[b["lane"] - 1]
        if c:  # 走数で縮める（20走ぶん場の平均を混ぜる）
            n, w = c[0], c[1]
            base = (n * w + 20 * base) / (n + 20)
        mres = num(b["mo"].get("残り"), 0.0)
        sres = num(b["se"].get("残り"), 0.0)
        s.append(math.log(max(base, 0.5)) - 0.6 * mres - 0.3 * sres)
    e = [math.exp(x) for x in s]
    t = sum(e)
    return [x / t for x in e]


def harville(p, a, b, c):
    pa = p[a]
    pb = p[b] / (1 - p[a])
    pc = p[c] / (1 - p[a] - p[b])
    return pa * pb * pc


def design(p, odds, head, budget_pts, lo):
    """頭を固定し、材料の確率が高い順に足していき、合成が lo を割る手前で止める。"""
    cands = []
    for b, c in itertools.permutations([i for i in range(6) if i != head], 2):
        key = f"{head+1}-{b+1}-{c+1}"
        if key in odds and odds[key]:
            cands.append((harville(p, head, b, c), key, odds[key]))
    cands.sort(reverse=True)
    picked = []
    for q, k, o in cands:
        trial = picked + [(q, k, o)]
        syn = 1 / sum(1 / x[2] for x in trial)
        if syn < lo and picked:
            continue
        picked = trial
        if len(picked) >= budget_pts:
            break
    if not picked:
        return None
    syn = 1 / sum(1 / x[2] for x in picked)
    hit = sum(x[0] for x in picked)
    return dict(points=picked, syn=syn, hit=hit, ev=hit * syn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--dossier", action="append", required=True, help="場名=jcd=資料のパス")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--lo", type=float, default=15)
    ap.add_argument("--hi", type=float, default=20)
    a = ap.parse_args()
    pts = a.budget // 100
    S = requests.Session()
    rows = []
    for spec in a.dossier:
        vname, jcd, path = spec.split("=", 2)
        d = load(path)
        if d["pday"] != a.date[4:]:
            print(f"skip {vname}: 番組が {d['pday']}")
            continue
        for r in d["prog"]:
            rno = int(r["R"])
            bs = [boat(d, i, r[str(i)]) for i in range(1, 7)]
            html = sw.fetch("odds3t", a.date, jcd, rno, S)
            odds = {c: v for _, c, v, _ in sw.parse_odds3t(html)} if html else {}
            if len(odds) < 100:
                print(f"{vname}{rno}R: オッズ未取得")
                continue
            p = win_scores(bs)
            inv = {h: sum(1 / v for k, v in odds.items() if v and k[0] == str(h + 1)) for h in range(6)}
            tot = sum(inv.values())
            m = [inv[h] / tot for h in range(6)]
            for h in range(6):
                dsg = design(p, odds, h, pts, a.lo)
                if not dsg:
                    continue
                rows.append(dict(v=vname, rno=rno, close=r["締切"], kind=r["種別"], head=h, name=bs[h]["name"],
                                 p=p[h], m=m[h], ratio=p[h] / max(m[h], 1e-6), p1=p[0], m1=m[0], **dsg))
    ok = [x for x in rows if a.lo <= x["syn"] <= a.hi * 1.5 and x["head"] != 0 and x["p"] >= 0.12]
    ok.sort(key=lambda x: (-x["ratio"]))
    best = {}
    for x in ok:
        best.setdefault((x["v"], x["rno"]), x)
    top = sorted(best.values(), key=lambda x: -x["ratio"])[:8]
    hd = f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}"
    L = [f"# 合成{a.lo:.0f}〜{a.hi:.0f}倍が組めそうなレース {hd}", ""]
    L.append(f"`scripts/analysis/value_scan.py` で作成。1レース{a.budget}円（100円×{pts}点まで）。オッズは展示前の締切前オッズ（確定とは違う、展示の後に動く）。")
    L.append("「材料の1着」＝選手のそのコースでの1着率（走数で場の平均に縮める）を、モーターと節間の『残り』で足し引きした目安。「市場の1着」＝3連単オッズから出した頭の見込み。比が1を超えるほど材料のわりに売れていない頭。検証していない目安。")
    L.append("並びは比（材料のわりに売れていない度合い）の順。この目安は1号艇を低く見積もりがち（全員A1級でどのコースの数字も高いため）なので、的中の確率や期待値としては使わない。")
    L.append("")
    L.append("| 順 | 場 | R | 締切 | 頭 | 材料の1着 | 市場の1着 | 比 | 1号艇（材料/市場） | 点数 | 合成 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, x in enumerate(top, 1):
        L.append(f"| {i} | {x['v']} | {x['rno']} | {x['close']} | {x['head']+1} {x['name']} | {x['p']:.0%} | {x['m']:.0%} | {x['ratio']:.2f} | {x['p1']:.0%}/{x['m1']:.0%} | {len(x['points'])} | {x['syn']:.1f} |")
    for i, x in enumerate(top, 1):
        L += ["", f"## {i}. {x['v']} {x['rno']}R（{x['close']}）頭 {x['head']+1} {x['name']}", ""]
        L.append("買い目の形（展示を見て直す前提のたたき台）: " + "、".join(f"{k} {o:.1f}倍" for _, k, o in x["points"]))
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    for i, x in enumerate(top, 1):
        print(i, x["v"], x["rno"], x["close"], x["head"] + 1, x["name"], f"p{x['p']:.2f} m{x['m']:.2f} r{x['ratio']:.2f} 1:{x['p1']:.2f}/{x['m1']:.2f} n{len(x['points'])} syn{x['syn']:.1f} hit{x['hit']:.2f} ev{x['ev']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
