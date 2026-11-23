"""3連単1点の知らせ（t3-lap-one-v1、notes/decisions.md 2026-11-24 に固定）。舟券は買わない。人が判断して買う。
条件（オッズは使わない）:
  1) オリジナル展示（BOATCAST）の一周タイム（桐生は半周）で、①が1位、②と③が2・3位
  2) 一周タイムの3位と4位の差が 0.4秒以上
  3) 買い目: ①頭の1点。2着・3着は②③のうち公式の展示タイムが速い方が2着（同タイムは②が先）
判定は展示とオリジナル展示が出た後（締切の数分前）。"""
from __future__ import annotations
import math
GAP34 = 0.4
NAME = "t3-lap-one-v1"


def judge(oriten_rows: list[dict], before_rows: list[dict]) -> dict | None:
    """oriten_rows: scrape_oriten.fetch の行（lane, lap）。before_rows: scrape_web.parse_beforeinfo の行（lane, exhibition_time）。
    条件に合えば {"combo": "1-2-3", "g34": 0.45, ...}、合わなければ None。材料が足りなければ {"wait": True}。"""
    ok = lambda v: v is not None and not (isinstance(v, float) and math.isnan(v))
    laps = {int(r["lane"]): float(r["lap"]) for r in oriten_rows if ok(r.get("lap"))}
    if len(laps) < 4:
        return {"wait": True}
    order = sorted(laps, key=lambda k: (laps[k], k))
    if order[0] != 1 or set(order[1:3]) != {2, 3}:
        return None
    g34 = round(laps[order[3]] - laps[order[2]], 2)
    if g34 < GAP34:
        return None
    et = {int(r["lane"]): r.get("exhibition_time") for r in before_rows if ok(r.get("exhibition_time"))}
    e2, e3 = et.get(2), et.get(3)
    if e2 is None or e3 is None:
        return {"wait": True}
    second, third = (2, 3) if e2 <= e3 else (3, 2)
    return {"combo": f"1-{second}-{third}", "g34": g34, "lap": [laps.get(k) for k in range(1, 7)], "ex2": e2, "ex3": e3}


def message(venue: str, rno: int, close: str, combo: str) -> str:
    return f"{venue}{rno}R（{close}締切）\n3連単 {combo}　100円\n※判断材料です。買うかどうかはご自身で。"
