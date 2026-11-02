"""公式ダウンロードサービス（番組表 B / 競走成績 K）から取り込んで SQLite に入れる。

boatrace-lzh ライブラリを使う（LZH 解凍に p7zip-full か lha が必要）。
番組表は翌日分まで先行公開されるので、前夜に翌日の出走表が取れる。

使い方:
  python scripts/ingest.py schedule --date 20261103          # 翌日の番組表
  python scripts/ingest.py performance --date 20261102       # 当日の成績
  python scripts/ingest.py performance --from 20231101 --to 20261101   # 3年分の一括取り込み
  python scripts/ingest.py schedule --from 20231101 --to 20261101

注意:
  - 成績（K）は boatrace-lzh の PerformanceParser を使う。項目名は 2026-11-01 の実データで確認済み
    （venue_code / race_number / boat_number / result_position / ticket_type / winning_combination / payout）。
  - 番組表（B）はライブラリのパーサーが全場1ファイルの実データに対応していないため parse_schedule_text() で自前パースする。
  - 一括取り込みは 1 日 2 ファイル × 日数。速さが要るときは月単位のプロセスを並列に回す（backfill スキル参照）。
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from datetime import timedelta

from common import CACHE_DIR, from_hd, to_hd
from db import connect, upsert


def _asdict(obj) -> dict:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _pick(d: dict, *names, default=None):
    """候補名のうち最初に見つかった値を返す。"""
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return default


_FW = str.maketrans("０１２３４５６７８９：", "0123456789:")
_RACE_HEAD = re.compile(r"^[\s　]*([０-９]{1,2})Ｒ\s*(.*?)\s*Ｈ([０-９]+)ｍ.*?電話投票締切予定([０-９]{1,2})：([０-９]{2})")
# 出走行の固定長カラム（全角1文字も1文字として数える）。列がくっつくことがあるので空白では区切らない。
#   例: "1 4208三浦永理43静岡45A1 7.21 60.48 6.58 38.46 24 28.87128 32.12             12"
_ENTRY_COLS = [
    ("national_win_rate", 18, 23, float), ("national_2ren_rate", 23, 29, float),
    ("local_win_rate", 29, 34, float), ("local_2ren_rate", 34, 40, float),
    ("motor_no", 40, 43, int), ("motor_2ren_rate", 43, 49, float),
    ("boat_no", 49, 52, int), ("boat_2ren_rate", 52, 58, float),
]


def _num(s: str):
    s = s.strip()
    if not s:
        return None
    try:
        return int(s) if s.isdigit() else float(s)
    except ValueError:
        return None


def parse_schedule_text(text: str) -> list[dict]:
    """番組表（B ファイル）を自前でパースする。

    boatrace-lzh 0.1.0 の ScheduleParser は 1 ファイル＝1 場を前提にしており、
    全場が 1 ファイルに入った実データ（"24BBGN"〜"24BEND" のブロックが場の数だけ並ぶ）では
    場コードが空になり、レース番号で上書きされて 12 レース分しか残らない。
    さらに出走行の勝率・モーター等も捨てているため、固定長の出走行をここで読む。
    （2026-11-01 の実データで確認）
    """
    rows: list[dict] = []
    jcd = None
    race = None
    for line in text.split("\n"):
        line = line.rstrip("\r")
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd, race = m.group(1), None
            continue
        if re.match(r"^\d{2}BEND", line):
            jcd, race = None, None
            continue
        if jcd is None:
            continue
        m = _RACE_HEAD.match(line)
        if m:
            race = {
                "rno": int(m.group(1).translate(_FW)),
                "race_name": m.group(2).replace("　", " ").strip(),
                "distance": int(m.group(3).translate(_FW)),
                "close_time": f"{int(m.group(4).translate(_FW)):02d}:{m.group(5).translate(_FW)}",
            }
            continue
        if race is None or not re.match(r"^[1-6] \d{4}", line):
            continue
        line = line.ljust(73)
        try:
            stats = {name: typ(line[i:j].strip()) for name, i, j, typ in _ENTRY_COLS}
        except ValueError:
            print(f"WARN: 出走行を読めない: {line!r}", file=sys.stderr)
            continue
        rows.append({
            "jcd": jcd, **race,
            "lane": int(line[0]),
            "racer_no": int(line[2:6]),
            "racer_name": line[6:10].replace("　", " ").strip(),
            "age": _num(line[10:12]),
            "branch": line[12:14].strip(),
            "weight": _num(line[14:16]),
            "grade": line[16:18].strip(),
            **stats,
            "series_results": line[59:71].rstrip(),
            "early_hayami": _num(line[71:73]),
        })
    return rows


def ingest_schedule(dl, d) -> int:
    files = dl.download(d, "schedule")
    if not files:
        return 0
    rows = []
    for text in files.values():
        for r in parse_schedule_text(text):
            rows.append({
                "hd": to_hd(d), "jcd": r["jcd"], "rno": r["rno"], "lane": r["lane"],
                "racer_no": r["racer_no"], "racer_name": r["racer_name"], "grade": r["grade"],
                "motor_no": r["motor_no"], "boat_no": r["boat_no"],
                "close_time": r["close_time"], "race_name": r["race_name"],
                "data": r,
            })
    con = connect()
    return upsert(con, "entries", rows)


_K_BLOCK = re.compile(r"^(\d{2})KBGN\s*$(.*?)^\d{2}KEND", flags=re.M | re.S)


def parse_performance_blocks(parser, files: dict[str, str]):
    """成績（K）ファイルを場ごとのブロック（"NNKBGN"〜"NNKEND"）に切ってからパースする。

    boatrace-lzh 0.1.0 の PerformanceParser は場名の部分一致で場を決めるため、
    「唐津」が「津」（09）と判定され、さらに場名をキーにまとめるので津と唐津が同じ日に
    開催すると片方が消える（2026-11-02 に判明）。場コードはブロック見出しの番号を正とする。
    """
    for name, text in files.items():
        for m in _K_BLOCK.finditer(text):
            yield m.group(1), parser.parse({f"{name}#{m.group(1)}": m.group(2)})


def ingest_performance(dl, parser, d) -> tuple[int, int]:
    files = dl.download(d, "performance")
    if not files:
        return 0, 0
    hd = to_hd(d)
    result_rows, payout_rows, race_rows = [], [], []
    for jcd, res in parse_performance_blocks(parser, files):
        for e in res.entries:
            ed = _asdict(e)
            ed["venue_code"] = jcd
            result_rows.append({
                "hd": hd, "jcd": jcd,
                "rno": int(ed["race_number"]),
                "lane": int(ed["boat_number"]),
                "position": ed.get("result_position"),
                "data": ed,
            })
        for p in res.payouts:
            pd_ = _asdict(p)
            payout_rows.append({
                "hd": hd, "jcd": jcd,
                "rno": int(pd_["race_number"]),
                "ticket_type": pd_["ticket_type"],
                "combination": str(pd_["winning_combination"]),
                "payout": int(pd_.get("payout") or 0),
            })
        for r in res.races:
            rd = _asdict(r)
            rd["venue_code"] = jcd
            race_rows.append({
                "hd": hd, "jcd": jcd, "rno": int(rd["race_number"]),
                "race_name": rd.get("race_name"), "distance": rd.get("distance"),
                "weather": rd.get("weather"), "wind_direction": rd.get("wind_direction"),
                "wind_speed": rd.get("wind_speed"), "wave_height": rd.get("wave_height"), "data": rd,
            })
    con = connect()
    upsert(con, "races", race_rows)
    return upsert(con, "results", result_rows), upsert(con, "payouts", payout_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["schedule", "performance"])
    ap.add_argument("--date")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    args = ap.parse_args()

    from boatrace_lzh import LzhDownloader, PerformanceParser

    if args.date:
        dates = [from_hd(args.date)]
    elif args.date_from and args.date_to:
        d0, d1 = from_hd(args.date_from), from_hd(args.date_to)
        dates = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    else:
        ap.error("--date か --from/--to を指定してください")
        return 2

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dl = LzhDownloader(cache_dir=str(CACHE_DIR), max_workers=2)
    parser = PerformanceParser()

    total = 0
    for d in dates:
        try:
            if args.kind == "schedule":
                n = ingest_schedule(dl, d)
                print(f"{d}: entries {n}")
                total += n
            else:
                n1, n2 = ingest_performance(dl, parser, d)
                print(f"{d}: results {n1}, payouts {n2}")
                total += n1
        except Exception as e:
            print(f"{d}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
    print(f"done: {total} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
