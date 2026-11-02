"""レース結果と払戻を取得して Supabase に保存する。

使い方:
  python scripts/fetch_results.py                 # 今日・全場
  python scripts/fetch_results.py --hd 20261102 --jcd 24 --rno 1
  python scripts/fetch_results.py --dry-run

TODO: セレクタは実ページで確認してから本実装。
"""
from __future__ import annotations

import argparse
import json
import sys

from bs4 import BeautifulSoup

from common import BASE_URL, VENUES, polite_get, supabase_client, today_hd


def parse_result(html: str, hd: str, jcd: str, rno: int) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select("div.table1 table")
    if not tables:
        return None
    # 着順表（1つ目）と払戻表（2つ目以降）を生テキストで保存。項目確定後に分解する。
    return {
        "hd": hd,
        "jcd": jcd,
        "rno": rno,
        "raw_order": [tr.get_text(" ", strip=True) for tr in tables[0].select("tbody tr")],
        "raw_payout": [tr.get_text(" ", strip=True) for t in tables[1:3] for tr in t.select("tbody tr")],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default=today_hd())
    ap.add_argument("--jcd")
    ap.add_argument("--rno", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    venues = [args.jcd] if args.jcd else list(VENUES)
    races = [args.rno] if args.rno else list(range(1, 13))

    results: list[dict] = []
    for jcd in venues:
        for rno in races:
            try:
                html = polite_get(f"{BASE_URL}/raceresult", {"rno": rno, "jcd": jcd, "hd": args.hd})
                row = parse_result(html, args.hd, jcd, rno)
            except Exception as e:
                print(f"skip {jcd} R{rno}: {e}", file=sys.stderr)
                continue
            if row:
                results.append(row)
                print(f"ok {VENUES[jcd]} R{rno}")

    if args.dry_run:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    sb = supabase_client()
    if results:
        sb.table("results").upsert(results, on_conflict="hd,jcd,rno").execute()
    print(f"saved {len(results)} races")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
