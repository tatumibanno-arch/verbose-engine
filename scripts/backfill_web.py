"""boatrace.jp の過去オッズ・直前情報を、SQLite を使わずに月単位で取り込む（GitHub Actions 用）。

この環境（Claude のクラウド）は会話が止まると機械ごと休止するので、長い取り込みは GitHub Actions で回す。
  fetch : results の Parquet から対象レースを選び、ページを取って data/web/<page>_<YYYYMM>.parquet に書く
          （途中で止まっても、同じファイルにある分は飛ばして続きから）
  import: data/web/*.parquet を SQLite の odds / beforeinfo に入れる（こちらの環境で実行）

使い方:
  python scripts/backfill_web.py fetch --month 202510 --pages odds2tf --workers 4 --max-minutes 320
  python scripts/backfill_web.py import
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from common import DATA_DIR, PARQUET_DIR

WEB_DIR = DATA_DIR / "web"
SAVE_EVERY = 500


def races_of_month(ym: str) -> list[tuple[str, str, int]]:
    path = PARQUET_DIR / f"results_{ym[:4]}.parquet"
    r = pd.read_parquet(path, columns=["hd", "jcd", "rno"])
    r = r[r.hd.str.startswith(ym)].drop_duplicates()
    return sorted((str(h), str(j).zfill(2), int(n)) for h, j, n in r.itertuples(index=False))


def out_path(page: str, ym: str) -> Path:
    return WEB_DIR / f"{page}_{ym}.parquet"


def to_rows(page: str, hd: str, jcd: str, rno: int, parsed) -> list[dict]:
    if page == "beforeinfo":
        return [{"hd": hd, "jcd": jcd, "rno": rno, "lane": r["lane"], "exhibition_time": r["exhibition_time"],
                 "tilt": r["tilt"], "start_course": r["start_course"], "start_st": r["start_st"],
                 "data": json.dumps(r, ensure_ascii=False)} for r in parsed]
    return [{"hd": hd, "jcd": jcd, "rno": rno, "ticket_type": tt, "combination": c, "odds": o, "odds_max": om}
            for tt, c, o, om in parsed]


def save(path: Path, old: pd.DataFrame | None, rows: list[dict], done_empty: set) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    df = new if old is None or old.empty else pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    if done_empty:  # 取ってもページが空だったレース（中止など）。二度取りしないよう別ファイルに残す
        ep = path.with_suffix(".empty.json")
        prev = set(map(tuple, json.loads(ep.read_text()))) if ep.exists() else set()
        ep.write_text(json.dumps(sorted(prev | done_empty)))
    return df


def cmd_fetch(ym: str, pages: list[str], workers: int, max_minutes: float) -> int:
    from scrape_web import work
    t_end = time.time() + max_minutes * 60
    races = races_of_month(ym)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers))
    for page in pages:
        path = out_path(page, ym)
        old = pd.read_parquet(path) if path.exists() else None
        done = set() if old is None else {(h, j, int(n)) for h, j, n in old[["hd", "jcd", "rno"]].drop_duplicates().itertuples(index=False)}
        ep = path.with_suffix(".empty.json")
        if ep.exists():
            done |= {(h, j, int(n)) for h, j, n in json.loads(ep.read_text())}
        todo = [r for r in races if r not in done]
        print(f"{page} {ym}: races {len(races)}, todo {len(todo)}", flush=True)
        rows, empty, n, t0 = [], set(), 0, time.time()
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(work, page, h, j, r, session): (h, j, r) for h, j, r in todo}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    _, hd, jcd, rno, parsed = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR {key}: {e}", file=sys.stderr, flush=True)
                    continue
                n += 1
                if parsed:
                    rows += to_rows(page, hd, jcd, rno, parsed)
                else:
                    empty.add(key)
                if n % SAVE_EVERY == 0:
                    old = save(path, old, rows, empty); rows, empty = [], set()
                    print(f"  {n}/{len(todo)} {n / (time.time() - t0):.2f}/s", flush=True)
                if time.time() > t_end:
                    print("時間切れ。ここまでを保存して終了（次の実行で続きから）", flush=True)
                    for f in futs:
                        f.cancel()
                    break
        save(path, old, rows, empty)
        print(f"{page} {ym}: saved ({n} pages this run)", flush=True)
        if time.time() > t_end:
            break
    return 0


def cmd_import() -> int:
    from db import connect, upsert
    con = connect()
    for path in sorted(WEB_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        table = "beforeinfo" if path.name.startswith("beforeinfo") else "odds"
        if table == "beforeinfo":
            df["data"] = df["data"].map(json.loads)
        upsert(con, table, df.to_dict("records"))
        print(f"{path.name}: {len(df)} rows → {table}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "import"])
    ap.add_argument("--month")
    ap.add_argument("--pages", default="odds2tf")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=320)
    a = ap.parse_args()
    if a.cmd == "fetch":
        return cmd_fetch(a.month, a.pages.split(","), a.workers, a.max_minutes)
    return cmd_import()


if __name__ == "__main__":
    raise SystemExit(main())
