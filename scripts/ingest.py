"""公式ダウンロードサービス（番組表 B / 競走成績 K）から取り込んで SQLite に入れる。

boatrace-lzh ライブラリを使う（LZH 解凍に p7zip-full か lha が必要）。
番組表は翌日分まで先行公開されるので、前夜に翌日の出走表が取れる。

使い方:
  python scripts/ingest.py schedule --date 20261103          # 翌日の番組表
  python scripts/ingest.py performance --date 20261102       # 当日の成績
  python scripts/ingest.py performance --from 20231101 --to 20261101   # 3年分の一括取り込み
  python scripts/ingest.py schedule --from 20231101 --to 20261101

注意:
  - 初回実行時に boatrace-lzh の各レコードの項目名を実データで確認し、下の _pick() の候補名を直すこと。
    どの項目が来ても data 列に JSON で丸ごと残しているので、取り直しは不要。
  - 一括取り込みは 1 日 2 ファイル × 日数。公式サーバーに負荷をかけないよう max_workers は 2 まで。
"""
from __future__ import annotations

import argparse
import dataclasses
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


def ingest_schedule(dl, parser, d) -> int:
    files = dl.download(d, "schedule")
    if not files:
        return 0
    res = parser.parse(files)
    races = {}
    for r in res.races:
        rd = _asdict(r)
        races[(str(_pick(rd, "venue_code", "jcd")).zfill(2), int(_pick(rd, "race_number", "rno")))] = rd
    rows = []
    for e in res.entries:
        ed = _asdict(e)
        jcd = str(_pick(ed, "venue_code", "jcd")).zfill(2)
        rno = int(_pick(ed, "race_number", "rno"))
        rd = races.get((jcd, rno), {})
        rows.append({
            "hd": to_hd(d), "jcd": jcd, "rno": rno,
            "lane": int(_pick(ed, "boat_number", "lane", "waku")),
            "racer_no": _pick(ed, "racer_number", "racer_no", "toban"),
            "racer_name": _pick(ed, "racer_name", "name"),
            "grade": _pick(ed, "grade", "class", "rank"),
            "motor_no": _pick(ed, "motor_number", "motor_no"),
            "boat_no": _pick(ed, "boat_id", "boat_no", "hull_number"),
            "close_time": _pick(rd, "close_time", "deadline"),
            "race_name": _pick(rd, "race_name", "title"),
            "data": ed,
        })
    con = connect()
    return upsert(con, "entries", rows)


def ingest_performance(dl, parser, d) -> tuple[int, int]:
    files = dl.download(d, "performance")
    if not files:
        return 0, 0
    res = parser.parse(files)
    hd = to_hd(d)
    result_rows = []
    for e in res.entries:
        ed = _asdict(e)
        result_rows.append({
            "hd": hd,
            "jcd": str(_pick(ed, "venue_code", "jcd")).zfill(2),
            "rno": int(_pick(ed, "race_number", "rno")),
            "lane": int(_pick(ed, "boat_number", "lane", "waku")),
            "position": _pick(ed, "result_position", "position", "rank"),
            "data": ed,
        })
    payout_rows = []
    for p in res.payouts:
        pd_ = _asdict(p)
        payout_rows.append({
            "hd": hd,
            "jcd": str(_pick(pd_, "venue_code", "jcd")).zfill(2),
            "rno": int(_pick(pd_, "race_number", "rno")),
            "ticket_type": _pick(pd_, "ticket_type", "type"),
            "combination": str(_pick(pd_, "winning_combination", "combination")),
            "payout": int(_pick(pd_, "payout", "amount", default=0)),
        })
    con = connect()
    return upsert(con, "results", result_rows), upsert(con, "payouts", payout_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["schedule", "performance"])
    ap.add_argument("--date")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    args = ap.parse_args()

    from boatrace_lzh import LzhDownloader, PerformanceParser, ScheduleParser

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
    parser = ScheduleParser() if args.kind == "schedule" else PerformanceParser()

    total = 0
    for d in dates:
        try:
            if args.kind == "schedule":
                n = ingest_schedule(dl, parser, d)
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
