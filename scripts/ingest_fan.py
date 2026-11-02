"""選手期別成績（公式「ファン手帳」データ fanYYMM.lzh）を racer_stats に取り込む。

半年ごと（4月・10月）に出る固定長ファイル。コース別の進入回数・着順回数・平均ST・複勝率など、
番組表に無い「1コース1着率」系の数字はここから作る。
レイアウト: https://www.boatrace.jp/owpc/pc/extra/data/layout.html（cp932・バイト固定長）

リーク防止: racer_stats.hd は算出期間の最終日。レース日より前に hd がある最新の行だけを使うこと。

使い方:
  python scripts/ingest_fan.py            # 公開されている全期を取り込む（取り込み済みは飛ばす）
  python scripts/ingest_fan.py --only 2510
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from common import CACHE_DIR
from db import connect, upsert

INDEX_URL = "https://www.boatrace.jp/owpc/pc/extra/data/download.html"
FILE_URL = "https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fan{}.lzh"

# (項目名, バイト数, 種類)  種類: s=文字列, i=整数, d1/d2=小数点以下1/2桁
_BASE = [
    ("racer_no", 4, "s"), ("name_kanji", 16, "s"), ("name_kana", 15, "s"), ("branch", 4, "s"),
    ("grade", 2, "s"), ("era", 1, "s"), ("birth", 6, "s"), ("sex", 1, "i"), ("age", 2, "i"),
    ("height", 3, "i"), ("weight", 2, "i"), ("blood", 2, "s"),
    ("win_rate", 4, "d2"), ("top2_rate", 4, "d1"), ("first_count", 3, "i"), ("second_count", 3, "i"),
    ("starts", 3, "i"), ("final_count", 2, "i"), ("champion_count", 2, "i"), ("avg_st", 3, "d2"),
]
_COURSE = [(f"c{c}_{n}", w, t) for c in range(1, 7) for n, w, t in [
    ("starts", 3, "i"), ("top2_rate", 4, "d1"), ("avg_st", 3, "d2"), ("avg_start_rank", 3, "d2")]]
_MID = [
    ("grade_prev1", 2, "s"), ("grade_prev2", 2, "s"), ("grade_prev3", 2, "s"),
    ("ability_prev", 4, "d2"), ("ability", 4, "d2"),
    ("year", 4, "i"), ("term", 1, "i"), ("period_from", 8, "s"), ("period_to", 8, "s"),
    ("training_term", 3, "i"),
]
_COURSE_POS = [(f"c{c}_{n}", w, "i") for c in range(1, 7) for n, w in [
    ("pos1", 3), ("pos2", 3), ("pos3", 3), ("pos4", 3), ("pos5", 3), ("pos6", 3),
    ("F", 2), ("L0", 2), ("L1", 2), ("K0", 2), ("K1", 2), ("S0", 2), ("S1", 2), ("S2", 2)]]
_TAIL = [("nocourse_L0", 2, "i"), ("nocourse_L1", 2, "i"), ("nocourse_K0", 2, "i"),
         ("nocourse_K1", 2, "i"), ("hometown", 6, "s")]
LAYOUT = _BASE + _COURSE + _MID + _COURSE_POS + _TAIL


def _val(raw: bytes, typ: str):
    s = raw.decode("cp932", errors="replace").replace("　", " ").strip()
    if typ == "s":
        return re.sub(r"\s+", " ", s) or None
    if not s.isdigit():
        return None
    n = int(s)
    return {"i": n, "d1": n / 10, "d2": n / 100}[typ]


def parse_fan(raw: bytes) -> list[dict]:
    rows = []
    for line in raw.split(b"\n"):
        line = line.rstrip(b"\r")
        if len(line) < 400:
            continue
        rec, pos = {}, 0
        for name, width, typ in LAYOUT:
            rec[name] = _val(line[pos:pos + width], typ)
            pos += width
        for c in range(1, 7):  # 使いやすいように率も出しておく（%）
            n = rec.get(f"c{c}_starts") or 0
            rec[f"c{c}_win_rate"] = round(100 * (rec.get(f"c{c}_pos1") or 0) / n, 1) if n else None
        rows.append(rec)
    return rows


def available() -> list[str]:
    html = requests.get(INDEX_URL, timeout=60).text
    return sorted(set(re.findall(r"kibetsu/fan(\d{4})\.lzh", html)))


def fetch(yymm: str) -> bytes:
    cache = CACHE_DIR / "fan" / f"fan{yymm}.txt"
    if cache.exists():
        return cache.read_bytes()
    r = requests.get(FILE_URL.format(yymm), timeout=120)
    r.raise_for_status()
    with tempfile.TemporaryDirectory() as td:
        lzh = Path(td) / f"fan{yymm}.lzh"
        lzh.write_bytes(r.content)
        subprocess.run(["7z", "x", "-y", f"-o{td}/out", str(lzh)], check=True, capture_output=True)
        txt = next(Path(td, "out").iterdir()).read_bytes()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(txt)
    return txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    args = ap.parse_args()
    con = connect()
    for yymm in ([args.only] if args.only else available()):
        try:
            recs = parse_fan(fetch(yymm))
        except Exception as e:
            print(f"fan{yymm}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
            continue
        rows = [{
            "hd": r["period_to"], "racer_no": r["racer_no"], "year": r["year"], "term": r["term"],
            "period_from": r["period_from"], "grade": r["grade"], "win_rate": r["win_rate"],
            "top2_rate": r["top2_rate"], "avg_st": r["avg_st"], "data": r,
        } for r in recs if r["racer_no"] and r["period_to"]]
        print(f"fan{yymm}: {upsert(con, 'racer_stats', rows)} racers "
              f"({rows[0]['period_from']}-{rows[0]['hd']})" if rows else f"fan{yymm}: 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
