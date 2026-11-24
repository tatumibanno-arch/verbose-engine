"""3連単1点の知らせ その2（t3-score-one-v1、notes/decisions.md 2026-11-24 に固定）。舟券は買わない。人が判断して買う。
艇ごとに7つの順位を足した「点数」（小さいほど良い）: 一周・まわり足・直線・公式展示タイム（速い順）＋全国勝率・モーター2連率・その枠の2連率（高い順）。
条件: ①の点数が1位。2・3着＝①以外で点数の良い2艇。その2艇（3番手）と4番手の点数の差が10以上。
買い目: ①頭の1点。2着・3着は選んだ2艇のうち公式展示タイムが速い方が2着。オッズは使わない。
過去の検証（scripts/analysis/one_pick.py）と同じ計算: 順位は同順位を平均、まわり足・直線が無い場は3.5、他の欠けは0として合計、一周タイムが5艇以上あるレースだけ。"""
from __future__ import annotations
import json, math, sqlite3
from pathlib import Path
GAP = 10
NAME = "t3-score-one-v1"
DB = Path(__file__).resolve().parent.parent / "data" / "boatrace.db"


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _rank(vals: dict, high_good=False) -> dict:
    """{lane: 値} → {lane: 順位}（同順位は平均。値が無い艇は入れない）"""
    items = [(k, v) for k, v in vals.items() if _ok(v)]
    items.sort(key=lambda kv: -kv[1] if high_good else kv[1])
    out, i = {}, 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][1] == items[i][1]:
            j += 1
        for t in range(i, j + 1):
            out[items[t][0]] = (i + j) / 2 + 1
        i = j + 1
    return out


def load_static(hd: str) -> dict:
    """{(jcd, rno): {lane: (全国勝率, モーター2連率, 枠の2連率)}}。期別成績はレース日より前に出たものだけ。"""
    con = sqlite3.connect(DB)
    ent = con.execute("select jcd, rno, lane, racer_no, data from entries where hd=?", (hd,)).fetchall()
    out: dict = {}
    for jcd, rno, lane, racer_no, data in ent:
        d = json.loads(data)
        row = con.execute("select data from racer_stats where racer_no=? and hd<? order by hd desc limit 1", (str(racer_no), hd)).fetchone()
        c2r = None
        if row:
            s = json.loads(row[0]); k = int(lane)
            n = s.get(f"c{k}_starts") or 0; t2 = (s.get(f"c{k}_pos1") or 0) + (s.get(f"c{k}_pos2") or 0)
            c2r = (t2 + 2) / (n + 5)
        out.setdefault((jcd, int(rno)), {})[int(lane)] = (d.get("national_win_rate"), d.get("motor_2ren_rate"), c2r)
    return out


def judge(oriten_rows: list[dict], before_rows: list[dict], static: dict) -> dict | None:
    lap = {int(r["lane"]): r.get("lap") for r in oriten_rows}; maw = {int(r["lane"]): r.get("mawari") for r in oriten_rows}
    cho = {int(r["lane"]): r.get("chokusen") for r in oriten_rows}
    et = {int(r["lane"]): r.get("exhibition_time") for r in before_rows}
    if sum(_ok(v) for v in lap.values()) < 5:
        return {"wait": True}
    if sum(_ok(v) for v in et.values()) < 5:
        return {"wait": True}
    lanes = range(1, 7)
    rl, rm, rc, re_ = _rank(lap), _rank(maw), _rank(cho), _rank(et)
    rn = _rank({k: (static.get(k) or (None,) * 3)[0] for k in lanes}, True)
    rmo = _rank({k: (static.get(k) or (None,) * 3)[1] for k in lanes}, True)
    rc2 = _rank({k: (static.get(k) or (None,) * 3)[2] for k in lanes}, True)
    sc = {k: rl.get(k, 0) + rm.get(k, 3.5) + rc.get(k, 3.5) + re_.get(k, 0) + rn.get(k, 0) + rmo.get(k, 0) + rc2.get(k, 0)
          for k in lanes if k in lap}
    order = sorted(sc, key=lambda k: (sc[k], k))
    if len(order) < 4 or order[0] != 1:
        return None
    gap = sc[order[3]] - sc[order[2]]
    if gap < GAP:
        return None
    a, b = order[1], order[2]
    ea, eb = et.get(a), et.get(b)
    second, third = (a, b) if (_ok(ea) and _ok(eb) and ea <= eb) else (b, a)
    return {"combo": f"1-{second}-{third}", "gap": round(gap, 1)}


GAP4 = 5.5
NAME4 = "t3-score4-v1"


def judge4(oriten_rows: list[dict], before_rows: list[dict], static: dict) -> dict | None:
    """3連単1点の知らせ その3 t3-score4-v1（2026-11-24 から LINE も送る）。点数＝一周・直線・公式展示タイム（速い順）＋モーター2連率（高い順）の順位の合計。
    ①が点数1位、3番手と4番手の差が5.5以上（過去の上位4%の線）。買い目は judge と同じ作り方（①頭、2・3着は公式展示順）。"""
    lap = {int(r["lane"]): r.get("lap") for r in oriten_rows}; cho = {int(r["lane"]): r.get("chokusen") for r in oriten_rows}
    et = {int(r["lane"]): r.get("exhibition_time") for r in before_rows}
    if sum(_ok(v) for v in lap.values()) < 5 or sum(_ok(v) for v in et.values()) < 5:
        return {"wait": True}
    rl, rc, re_ = _rank(lap), _rank(cho), _rank(et)
    rmo = _rank({k: (static.get(k) or (None,) * 3)[1] for k in range(1, 7)}, True)
    sc = {k: rl.get(k, 0) + rc.get(k, 3.5) + re_.get(k, 0) + rmo.get(k, 0) for k in range(1, 7) if k in lap}
    order = sorted(sc, key=lambda k: (sc[k], k))
    if len(order) < 4 or order[0] != 1 or sc[order[3]] - sc[order[2]] < GAP4:
        return None
    a, b = order[1], order[2]; ea, eb = et.get(a), et.get(b)
    second, third = (a, b) if (_ok(ea) and _ok(eb) and ea <= eb) else (b, a)
    return {"combo": f"1-{second}-{third}", "gap": round(sc[order[3]] - sc[order[2]], 1)}
