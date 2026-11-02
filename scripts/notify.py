"""選定結果・検証結果を通知する。通知先は会話で決めてから実装する。

使い方:
  python scripts/notify.py --mode morning   # 今日の選定レース
  python scripts/notify.py --mode evening   # 今日の検証結果

TODO: Slack Webhook か Notion かを決める。決まるまで標準出力に出すだけ。
"""
from __future__ import annotations

import argparse
import json
import os

import requests

from common import VENUES, supabase_client, today_hd


def build_morning(hd: str) -> str:
    sb = supabase_client()
    picks = sb.table("picks").select("*").eq("hd", hd).order("jcd").order("rno").execute().data
    if not picks:
        return f"{hd} 選定なし"
    lines = [f"🚤 {hd} 選定 {len(picks)}レース"]
    for p in picks:
        lines.append(f"- {VENUES[p['jcd']]} {p['rno']}R {p['bet_type']} {p['pattern']}")
    return "\n".join(lines)


def build_evening(hd: str) -> str:
    sb = supabase_client()
    v = sb.table("verifications").select("*").eq("hd", hd).execute().data
    if not v:
        return f"{hd} 検証なし"
    v = v[0]
    return (
        f"📊 {hd} 検証\n"
        f"対象 {v['races']} / 的中 {v['hits']} / 的中率 {v['hit_rate']}\n"
        f"購入 ¥{v['bet_total']} / 払戻 ¥{v['payout_total']} / 回収率 {v['roi']}"
    )


def send(text: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        requests.post(webhook, json={"text": text}, timeout=10).raise_for_status()
    else:
        print(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["morning", "evening"], required=True)
    ap.add_argument("--hd", default=today_hd())
    args = ap.parse_args()
    text = build_morning(args.hd) if args.mode == "morning" else build_evening(args.hd)
    send(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
