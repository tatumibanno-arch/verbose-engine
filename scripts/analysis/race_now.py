"""締切前に1レースの展示・オッズを取り、指定した買い目に1,000円を配る（運用者の遊び用、舟券は買わない）。

  python scripts/analysis/race_now.py --jcd 15 --rno 5 --date 20261116            # 展示と人気・頭別のオッズ
  python scripts/analysis/race_now.py --jcd 15 --rno 5 --date 20261116 --combos 3-1-4,3-4-1,... --budget 1000
配分は「どれが当たっても払戻がだいたい同じ」（1点 ≒ 予算 × 合成 ÷ オッズ、100円単位、最低100円）。オッズは締切前のもの。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scrape_web as sw  # noqa: E402


def allocate(odds: dict, combos: list[str], budget: int):
    v = [odds[c] for c in combos]
    syn = 1 / sum(1 / x for x in v)
    raw = [budget * syn / x for x in v]
    st = [max(100, int(round(r / 100)) * 100) for r in raw]
    # 予算に合わせて、払戻が一番大きい目から削る／一番小さい目に足す
    while sum(st) > budget:
        i = max((i for i in range(len(st)) if st[i] > 100), key=lambda i: st[i] * v[i], default=None)
        if i is None:
            break
        st[i] -= 100
    while sum(st) < budget:
        i = min(range(len(st)), key=lambda i: st[i] * v[i])
        st[i] += 100
    return syn, list(zip(combos, v, st))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jcd", required=True)
    ap.add_argument("--rno", type=int, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--combos", default="")
    ap.add_argument("--budget", type=int, default=1000)
    a = ap.parse_args()
    S = requests.Session()
    b = sw.parse_beforeinfo(sw.fetch("beforeinfo", a.date, a.jcd, a.rno, S) or "")
    if b and b[0].get("exhibition_time"):
        ex = sorted(b, key=lambda r: r["exhibition_time"])
        rank = {r["lane"]: i + 1 for i, r in enumerate(ex)}
        for r in b:
            print(f"{r['lane']}号艇 展示{r['exhibition_time']}({rank[r['lane']]}位) 進入{r.get('start_course')} スタ展{r.get('start_st_raw')} チルト{r.get('tilt')} 部品{r.get('parts_changed') or '-'}")
        print(f"風 {b[0].get('wind_speed')}m 向き{b[0].get('wind_direction')} 波{b[0].get('wave_height')}cm")
    else:
        print("展示まだ")
    odds = {c: v for _, c, v, _ in sw.parse_odds3t(sw.fetch("odds3t", a.date, a.jcd, a.rno, S) or "") if v}
    if not odds:
        print("オッズ取れず")
        return 1
    inv = {h: sum(1 / v for k, v in odds.items() if k[0] == h) for h in "123456"}
    t = sum(inv.values())
    print("市場の頭: " + " ".join(f"{h}:{inv[h]/t:.0%}" for h in "123456"))
    print("人気上位: " + " ".join(f"{k} {v}" for k, v in sorted(odds.items(), key=lambda x: x[1])[:10]))
    for h in "123456":
        print(f"頭{h}: " + " ".join(f"{k} {v}" for k, v in sorted([(k, v) for k, v in odds.items() if k[0] == h], key=lambda x: x[1])[:10]))
    if a.combos:
        cs = [c.strip() for c in a.combos.split(",") if c.strip()]
        syn, rows = allocate(odds, cs, a.budget)
        print(f"\n合成 {syn:.1f}倍（{len(cs)}点）")
        for c, v, s in rows:
            print(f"{c} {s}円（{v}倍→{int(s*v):,}円）")
        print(f"計 {sum(s for _,_,s in rows):,}円 / 最低払戻 {min(int(s*v) for _,v,s in rows):,}円")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
