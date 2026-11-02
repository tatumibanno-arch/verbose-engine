"""出走表を取得して Supabase に保存する。

使い方:
  python scripts/fetch_racelist.py                # 今日・全場
  python scripts/fetch_racelist.py --hd 20261102 --jcd 24 --rno 1   # 1レースだけ（動作確認用）
  python scripts/fetch_racelist.py --dry-run      # 保存せず標準出力に出す

TODO（会話で決める）:
  - 取得項目の確定（現状は選手・級別・モーター番号・2連率の最低限）
  - 当日開催場の自動判定（現状は --jcd 指定 or VENUES 全走査）
"""
from __future__ import annotations

import argparse
import json
import sys

from bs4 import BeautifulSoup

from common import BASE_URL, VENUES, polite_get, supabase_client, today_hd


def parse_racelist(html: str, hd: str, jcd: str, rno: int) -> list[dict]:
    """出走表ページから1艇1行の dict を返す。セレクタは実ページで要検証。"""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    tbodies = soup.select("div.table1 table tbody.is-fs12")
    for lane, tbody in enumerate(tbodies, start=1):
        tds = tbody.find_all("td")
        text = [td.get_text(" ", strip=True) for td in tds]
        rows.append({
            "hd": hd,
            "jcd": jcd,
            "rno": rno,
            "lane": lane,
            "raw": text,          # まずは生テキストを保存し、項目確定後に分解する
        })
    return rows


def fetch_race(hd: str, jcd: str, rno: int) -> list[dict]:
    html = polite_get(f"{BASE_URL}/racelist", {"rno": rno, "jcd": jcd, "hd": hd})
    return parse_racelist(html, hd, jcd, rno)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default=today_hd())
    ap.add_argument("--jcd", help="場コード 01-24。省略時は全場")
    ap.add_argument("--rno", type=int, help="レース番号 1-12。省略時は全レース")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    venues = [args.jcd] if args.jcd else list(VENUES)
    races = [args.rno] if args.rno else list(range(1, 13))

    all_rows: list[dict] = []
    for jcd in venues:
        for rno in races:
            try:
                rows = fetch_race(args.hd, jcd, rno)
            except Exception as e:  # 非開催場は404等で落ちるので握って続行
                print(f"skip {jcd} R{rno}: {e}", file=sys.stderr)
                continue
            if rows:
                all_rows.extend(rows)
                print(f"ok {VENUES[jcd]} R{rno}: {len(rows)}艇")

    if args.dry_run:
        print(json.dumps(all_rows, ensure_ascii=False, indent=2))
        return 0

    sb = supabase_client()
    if all_rows:
        sb.table("racelist").upsert(all_rows, on_conflict="hd,jcd,rno,lane").execute()
    print(f"saved {len(all_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
