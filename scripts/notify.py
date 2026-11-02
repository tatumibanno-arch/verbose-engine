"""選定結果・検証結果を通知する。SLACK_WEBHOOK_URL が無ければ標準出力に出すだけ。

使い方:
  python scripts/notify.py --mode picks      # 翌日の買い目
  python scripts/notify.py --mode verify     # 今日の検証
"""
from __future__ import annotations

import argparse
import os

import requests

from common import to_hd, today, tomorrow, venue_name
from db import connect


def build_picks(hd: str) -> str:
    con = connect()
    rows = con.execute(
        "select jcd, rno, ticket_type, combination, amount from picks where hd=? order by jcd, rno, combination",
        (hd,)).fetchall()
    if not rows:
        return f"🚤 {hd} 選定なし"
    total = sum(r[4] for r in rows)
    lines = [f"🚤 {hd} 買い目 {len(rows)}点 / ¥{total}"]
    cur = None
    for jcd, rno, tt, combo, amt in rows:
        if (jcd, rno) != cur:
            lines.append(f"\n{venue_name(jcd)} {rno}R")
            cur = (jcd, rno)
        lines.append(f"  {combo} ¥{amt}")
    return "\n".join(lines)


def build_verify(hd: str) -> str:
    con = connect()
    r = con.execute("select races, bets, hits, bet_total, payout_total, hit_rate, roi from verifications where hd=?",
                    (hd,)).fetchone()
    if not r:
        return f"📊 {hd} 検証なし"
    return (f"📊 {hd} 検証\n{r[0]}レース {r[1]}点 / 的中 {r[2]} ({r[5]})\n"
            f"購入 ¥{r[3]} / 払戻 ¥{r[4]} / 回収率 {r[6]}")


def send(text: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        requests.post(webhook, json={"text": text}, timeout=10).raise_for_status()
    print(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["picks", "verify"], required=True)
    ap.add_argument("--date")
    args = ap.parse_args()
    if args.mode == "picks":
        send(build_picks(args.date or to_hd(tomorrow())))
    else:
        send(build_verify(args.date or to_hd(today())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
