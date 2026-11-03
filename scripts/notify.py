"""選定結果・検証結果を通知する。

送り先（設定があるものすべてに送る。どれも無ければ標準出力に出すだけ）:
  * LINE: 環境変数 LINE_USER_ID（U から始まる送り先ID）。トークンは LINE_CHANNEL_TOKEN、
          またはネットワークシークレットで api.line.me に Authorization を付ける設定。
  * Slack: 環境変数 SLACK_WEBHOOK_URL

使い方:
  python scripts/notify.py --mode picks      # 翌日の買い目
  python scripts/notify.py --mode verify     # 今日の検証
"""
from __future__ import annotations

import argparse
import os

import requests

from common import load_rules, to_hd, today, tomorrow, venue_name
from db import connect
from select_races import rule_of


LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_TEXT_MAX = 4900        # LINE のテキスト1通の上限は5000文字
LINE_MESSAGES_MAX = 5       # push 1回で送れるのは5通まで


def build_picks_line(hd: str) -> str:
    """スマホで読みやすい形: 締切順に1レース1行、買い目はまとめて書く。"""
    con = connect()
    rows = con.execute(
        "select p.jcd, p.rno, p.ticket_type, p.combination, p.amount, p.reason, "
        "(select min(e.close_time) from entries e where e.hd=p.hd and e.jcd=p.jcd and e.rno=p.rno) "
        "from picks p where p.hd=? order by 7, p.jcd, p.rno, p.combination", (hd,)).fetchall()
    day = f"{int(hd[4:6])}/{int(hd[6:])}"
    if not rows:
        return f"🚤 {day} の買い目はありません（条件に合うレースなし）"
    names = {"tansho": "単勝", "fukusho": "複勝", "nirentan": "2連単", "nirensho": "2連単",
             "sanrentan": "3連単", "sanrensho": "3連単"}
    by_rule: dict[str, dict] = {}
    for jcd, rno, tt, combo, amt, reason, close in rows:
        races = by_rule.setdefault(rule_of(reason), {})
        races.setdefault((close or "", jcd, rno, tt), []).append((combo, amt))
    summary_only = set((load_rules().get("notify") or {}).get("summary_only_rules") or [])
    total = sum(r[4] for r in rows)
    out = [f"🚤 {day} の買い目 {len(rows)}点 / {total:,}円", "（朝8時にまとめて購入。詳細は notes/picks/）"]
    for rule, races in by_rule.items():
        n = sum(len(v) for v in races.values())
        out.append(f"\n■ {rule}  {len(races)}レース {n}点 / {sum(a for v in races.values() for _, a in v):,}円")
        if rule in summary_only:
            out.append("（比較用のルール。買い目は notes/picks/ に）")
            continue
        for (close, jcd, rno, tt), combos in races.items():
            amt = {a for _, a in combos}
            each = (f" {amt.pop()}円" if len(combos) == 1 else f" 各{amt.pop()}円") if len(amt) == 1 else ""
            out.append(f"{close} {venue_name(jcd)}{rno}R {names.get(tt, tt)} {' '.join(c for c, _ in combos)}{each}")
    return "\n".join(out)


def build_picks(hd: str) -> str:
    con = connect()
    rows = con.execute(
        "select jcd, rno, ticket_type, combination, amount, reason from picks where hd=? order by jcd, rno, combination",
        (hd,)).fetchall()
    if not rows:
        return f"🚤 {hd} 選定なし"
    total = sum(r[4] for r in rows)
    lines = [f"🚤 {hd} 買い目 {len(rows)}点 / ¥{total}（詳細は notes/picks/）"]
    by_rule: dict[str, list] = {}
    for r in rows:
        by_rule.setdefault(rule_of(r[5]), []).append(r)
    for rule, rs in by_rule.items():
        lines.append(f"\n■ {rule} {len(rs)}点 / ¥{sum(r[4] for r in rs)}")
        cur = None
        for jcd, rno, tt, combo, amt, _ in rs:
            if (jcd, rno) != cur:
                lines.append(f"{venue_name(jcd)} {rno}R")
                cur = (jcd, rno)
            lines.append(f"  {tt} {combo} ¥{amt}")
    return "\n".join(lines)


def build_verify(hd: str) -> str:
    con = connect()
    r = con.execute("select races, bets, hits, bet_total, payout_total, hit_rate, roi from verifications where hd=?",
                    (hd,)).fetchone()
    if not r:
        return f"📊 {hd} 検証なし"
    text = (f"📊 {hd} 検証\n{r[0]}レース {r[1]}点 / 的中 {r[2]} ({r[5]})\n"
            f"購入 ¥{r[3]} / 払戻 ¥{r[4]} / 回収率 {r[6]}")
    for rule, bets, hits, bet, pay, roi in con.execute(
            "select rule, bets, hits, bet_total, payout_total, roi from verifications_by_rule where hd=? order by rule",
            (hd,)).fetchall():
        text += f"\n・{rule}: {bets}点 的中{hits} ¥{bet}→¥{pay} 回収率 {roi}"
    return text


def _chunks(text: str, size: int) -> list[str]:
    """行の途中で切らずに size 文字以内に分ける。"""
    out, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + 1 + len(line) > size:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def send_line(text: str) -> bool:
    to = os.environ.get("LINE_USER_ID")
    if not to:
        return False
    parts = _chunks(text, LINE_TEXT_MAX)
    if len(parts) > LINE_MESSAGES_MAX:
        parts = parts[:LINE_MESSAGES_MAX - 1] + ["…長いので残りは notes/picks/ を見てください"]
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(LINE_PUSH_URL, headers=headers, timeout=15,
                      json={"to": to, "messages": [{"type": "text", "text": t} for t in parts]})
    if r.status_code != 200:
        raise RuntimeError(f"LINE push failed: HTTP {r.status_code} {r.text[:200]}")
    return True


def send(text: str, line_text: str | None = None) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        requests.post(webhook, json={"text": text}, timeout=10).raise_for_status()
    try:
        if send_line(line_text or text):
            print("(LINE に送信しました)")
    except Exception as e:  # 通知の失敗で夜の処理（コミットなど）を止めない
        print(f"WARN LINE 送信に失敗: {e}")
    print(line_text or text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["picks", "verify"], required=True)
    ap.add_argument("--date")
    ap.add_argument("--dry-run", action="store_true", help="送らずに LINE 用の文面を表示するだけ")
    args = ap.parse_args()
    if args.dry_run:
        hd = args.date or to_hd(tomorrow() if args.mode == "picks" else today())
        print(build_picks_line(hd) if args.mode == "picks" else build_verify(hd))
        return 0
    if args.mode == "picks":
        hd = args.date or to_hd(tomorrow())
        send(build_picks(hd), build_picks_line(hd))
    else:
        send(build_verify(args.date or to_hd(today())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
