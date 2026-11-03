"""締切前アラート（strategies.live_tansho_poolgap_v1）。舟券は買わない。知らせるだけ。

しくみ:
  前夜  prepare: 翌日のレース一覧（締切時刻・選手名・モデルの1着確率）を data/live/<日付>.csv に書く（nightly が実行）
  当日  run:     締切 lead_minutes 分前に単勝・3連単のオッズを取り、
                 3連単から出した1着確率 ÷ 単勝から出した1着確率 ≥ min_gap かつ 3連単からの1着確率 ≥ min_p3
                 の艇を LINE に流す。全レースのオッズと判定を data/live/log/<日付>_s<枠>.csv に残す（後で検証する材料）
  夜    report:  ログと払戻を突き合わせて紙上の成績を出す（nightly が実行）

使い方:
  python scripts/live_alert.py prepare --date 20261104
  python scripts/live_alert.py run                # 今の時刻から受け持ちの枠（rules.yaml slots）を自動で選ぶ
  python scripts/live_alert.py run --test         # 次に締切が来るレースを1つだけ判定して、テストとして LINE に送る
  python scripts/live_alert.py report --date 20261104
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import requests

from common import DATA_DIR, JST, VENUES, load_rules, to_hd, today, tomorrow
from line_push import send_line

RULE = "live_tansho_poolgap_v1"
LIVE_DIR = DATA_DIR / "live"
LOG_DIR = LIVE_DIR / "log"
POLL_SECONDS = 20
LOG_FIELDS = ["hd", "jcd", "rno", "close", "checked_at", "lane", "name", "tansho_odds", "p_tansho", "p_3t",
              "gap", "p_model", "alert", "sent"]


def cfg() -> dict:
    return load_rules()["strategies"][RULE]


def now_jst() -> datetime:
    return datetime.now(JST)


def hm_to_dt(day: datetime, hm: str) -> datetime:
    h, m = map(int, hm.split(":"))
    return day.replace(hour=h, minute=m, second=0, microsecond=0)


# ------------------------------------------------------------------ prepare（前夜）
def cmd_prepare(hd: str) -> int:
    import pandas as pd

    from db import connect
    con = connect()
    e = pd.read_sql("select jcd, rno, lane, racer_name, close_time, race_name from entries where hd=?", con, params=(hd,))
    if e.empty:
        print(f"{hd}: 番組表がありません", file=sys.stderr)
        return 1
    races = e.groupby(["jcd", "rno"]).agg(close=("close_time", "first"), race_name=("race_name", "first")).reset_index()
    for lane in range(1, 7):
        nm = e[e.lane == lane].set_index(["jcd", "rno"]).racer_name
        races[f"name_{lane}"] = [nm.get((j, r), "") for j, r in zip(races.jcd, races.rno)]
    hist = Path(DATA_DIR / "models" / "tansho_ev" / "history.parquet")
    if hist.exists():
        h = pd.read_parquet(hist)
        h = h[h.hd == hd].drop_duplicates(["jcd", "rno"], keep="last").set_index(["jcd", "rno"])
        for lane in range(1, 7):
            races[f"p_{lane}"] = [h[f"p1r_{lane}"].get((j, r)) if (j, r) in h.index else None
                                  for j, r in zip(races.jcd, races.rno)]
    races = races.sort_values(["close", "jcd", "rno"])
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = LIVE_DIR / f"{hd}.csv"
    races.to_csv(out, index=False)
    print(f"{out}: {len(races)} races")
    return 0


def load_races(hd: str) -> list[dict]:
    path = LIVE_DIR / f"{hd}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["jcd"] = str(r["jcd"]).zfill(2)
        r["rno"] = int(r["rno"])
    return rows


# ------------------------------------------------------------------ 判定
def market_probs(hd: str, jcd: str, rno: int, session: requests.Session):
    """(単勝オッズ[6], 単勝由来の1着確率[6], 3連単由来の1着確率[6])。取れなければ None。"""
    from scrape_web import fetch, parse_odds3t, parse_oddstf
    tf = parse_oddstf(fetch("oddstf", hd, jcd, rno, session) or "")
    t3 = parse_odds3t(fetch("odds3t", hd, jcd, rno, session) or "")
    if not tf or not t3:
        return None
    odds = [None] * 6
    for tt, comb, o, _ in tf:
        if tt == "tansho":
            odds[int(comb) - 1] = o
    inv = [1 / o if o and o > 0 else 0.0 for o in odds]
    s = sum(inv)
    p_tan = [v / s if s else 0.0 for v in inv]
    inv3 = [0.0] * 6
    for _, comb, o, _ in t3:
        if o and o > 0:
            inv3[int(comb[0]) - 1] += 1 / o
    s3 = sum(inv3)
    p3 = [v / s3 if s3 else 0.0 for v in inv3]
    return odds, p_tan, p3


def judge(race: dict, hd: str, c: dict, session: requests.Session, checked_at: datetime):
    res = market_probs(hd, race["jcd"], race["rno"], session)
    rows = []
    if res is None:
        return rows
    odds, p_tan, p3 = res
    for i in range(6):
        gap = p3[i] / p_tan[i] if p_tan[i] > 0 else None
        alert = bool(odds[i] and odds[i] > 0 and gap is not None and gap >= c["min_gap"] and p3[i] >= c["min_p3"])
        pm = race.get(f"p_{i + 1}")
        rows.append({
            "hd": hd, "jcd": race["jcd"], "rno": race["rno"], "close": race["close"],
            "checked_at": checked_at.strftime("%H:%M:%S"), "lane": i + 1, "name": race.get(f"name_{i + 1}", ""),
            "tansho_odds": odds[i], "p_tansho": round(p_tan[i], 4), "p_3t": round(p3[i], 4),
            "gap": round(gap, 3) if gap is not None else None, "p_model": pm or None,
            "alert": int(alert), "sent": 0,
        })
    return rows


def message(race: dict, rows: list[dict], test: bool) -> str:
    head = "🧪【テスト】" if test else "🚤【紙上】"
    v = VENUES.get(race["jcd"], race["jcd"])
    lines = [f"{head}{race['close']}締切 {v}{race['rno']}R"]
    hits = [r for r in rows if r["alert"]]
    show = hits or ([max(rows, key=lambda r: r["gap"] or 0)] if test else [])
    for r in show:
        lines.append(f"単勝 {r['lane']}号艇 {r['name']}  オッズ {r['tansho_odds']}倍（{r['checked_at'][:5]}時点）")
        lines.append(f"3連単から見た1着率 {r['p_3t']:.0%} / 単勝の人気 {r['p_tansho']:.0%}（ずれ {r['gap']:.1f}倍）")
    if test and not hits:
        lines.append("※条件（ずれ・確率の下限）には届いていない。テスト表示です")
    return "\n".join(lines)


def append_log(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ run（当日）
def pick_slot(c: dict, now: datetime) -> int | None:
    for i, (a, b) in reversed(list(enumerate(c["slots"]))):  # 枠が重なる時間は後の枠を選ぶ
        if hm_to_dt(now, a) - timedelta(minutes=45) <= now < hm_to_dt(now, b):
            return i
    return None


def cmd_run(slot: int | None, test: bool, dry: bool) -> int:
    c = cfg()
    now = now_jst()
    hd = to_hd(today())
    races = load_races(hd)
    if not races:
        msg = f"⚠️ {hd} の締切前アラート: レース一覧（data/live/{hd}.csv）がありません。前夜の処理を確認してください"
        print(msg)
        if not dry and slot in (None, 0):
            send_line(msg)
        return 1
    session = requests.Session()

    if test:
        later = [r for r in races if hm_to_dt(now, r["close"]) > now + timedelta(minutes=1)]
        if not later:
            print("今日はもう締切前のレースがありません")
            return 0
        race = later[0]
        rows = judge(race, hd, c, session, now_jst())
        text = message(race, rows, test=True) if rows else f"🧪【テスト】{race['close']} {race['jcd']} {race['rno']}R オッズが取れませんでした"
        print(text)
        if not dry:
            send_line(text)
        return 0

    if slot is None:
        slot = pick_slot(c, now)
    if slot is None:
        print("受け持ちの枠がない時刻です")
        return 0
    s_from, s_to = c["slots"][slot]
    mine = [r for r in races if s_from <= r["close"] < s_to]
    win_from, win_to = c["alert_window"]
    log_path = LOG_DIR / f"{hd}_s{slot}.csv"
    done: set = set()
    if log_path.exists():  # 再実行したときに同じレースを二重に知らせない
        with open(log_path, encoding="utf-8") as f:
            done = {(r["jcd"], int(r["rno"])) for r in csv.DictReader(f)}
    end = hm_to_dt(now, s_to) + timedelta(minutes=2)
    print(f"{hd} slot {slot} {s_from}-{s_to}: races {len(mine)} (done {len(done)})", flush=True)
    n_alert = 0
    with ThreadPoolExecutor(4) as ex:
        while now_jst() < end:
            t = now_jst()
            due = []
            for r in mine:
                key = (r["jcd"], r["rno"])
                if key in done:
                    continue
                left = (hm_to_dt(t, r["close"]) - t).total_seconds() / 60
                if left < c["min_lead_minutes"]:
                    done.add(key)          # 間に合わない（ジョブの開始が遅れたなど）
                elif left <= c["lead_minutes"]:
                    due.append(r)
            if not due:
                if all((r["jcd"], r["rno"]) in done for r in mine):
                    break
                time.sleep(POLL_SECONDS)
                continue
            results = list(ex.map(lambda r: (r, judge(r, hd, c, session, now_jst())), due))
            for race, rows in results:
                done.add((race["jcd"], race["rno"]))
                if not rows:
                    print(f"  {race['close']} {race['jcd']} {race['rno']}R: オッズ取得失敗", flush=True)
                    continue
                if any(r["alert"] for r in rows) and win_from <= race["close"] < win_to:
                    text = message(race, rows, test=False)
                    print(text, flush=True)
                    if not dry:
                        try:
                            if send_line(text):
                                for r in rows:
                                    r["sent"] = r["alert"]
                        except Exception as e:  # 1通の失敗で全体を止めない
                            print(f"WARN LINE: {e}", flush=True)
                    n_alert += 1
                if not dry:
                    append_log(log_path, rows)
    print(f"slot {slot} done: alerts {n_alert}", flush=True)
    return 0


# ------------------------------------------------------------------ report（夜）
def cmd_report(hd: str) -> int:
    import pandas as pd

    from db import connect
    logs = sorted(LOG_DIR.glob(f"{hd}_s*.csv"))
    if not logs:
        print(f"{hd}: ログなし")
        return 0
    d = pd.concat([pd.read_csv(p, dtype={"jcd": str}) for p in logs])
    d["jcd"] = d.jcd.str.zfill(2)
    con = connect()
    pay = pd.read_sql("select jcd, rno, combination, payout from payouts where hd=? and ticket_type='tansho'",
                      con, params=(hd,))
    pay["lane"] = pay.combination.astype(int)
    d = d.merge(pay[["jcd", "rno", "lane", "payout"]], on=["jcd", "rno", "lane"], how="left")
    d["ret"] = d.payout.fillna(0)
    a = d[d.alert == 1]
    c = cfg()
    lines = [f"# 締切前アラート {hd}（{RULE}、紙上）", "",
             f"- 判定したレース {d.groupby(['jcd', 'rno']).ngroups} / アラート {len(a)}点（送信 {int(a.sent.sum())}）",
             f"- 的中 {int((a.ret > 0).sum())} / 購入 {len(a) * c['unit_yen']}円 / 払戻 {int(a.ret.sum() * c['unit_yen'] / 100)}円"
             + (f" / 回収率 {a.ret.sum() / (len(a) * 100):.2f}" if len(a) else ""), ""]
    for r in a.sort_values("close").itertuples():
        lines.append(f"- {r.close} {VENUES.get(r.jcd, r.jcd)}{r.rno}R {r.lane}号艇 {r.name} "
                     f"{r.tansho_odds}倍 ずれ{r.gap} → {'◎' + str(int(r.ret)) + '円' if r.ret > 0 else '×'}")
    text = "\n".join(lines)
    out = Path(__file__).resolve().parent.parent / "notes" / "live" / f"{hd}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "report"])
    ap.add_argument("--date")
    ap.add_argument("--slot", type=int)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        return cmd_prepare(a.date or to_hd(tomorrow()))
    if a.cmd == "report":
        return cmd_report(a.date or to_hd(today()))
    return cmd_run(a.slot, a.test, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
