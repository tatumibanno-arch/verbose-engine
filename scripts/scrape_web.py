"""boatrace.jp から締切時オッズと直前情報を取り込む。

公式ダウンロード（番組表 B / 成績 K）に無い情報を補う:
  - odds3t   : 3連単オッズ（120通り）
  - odds2tf  : 2連単（30通り）・2連複（15通り）オッズ
  - oddstf   : 単勝・複勝オッズ（複勝は下限〜上限）
  - beforeinfo: 直前情報（体重・展示タイム・チルト・プロペラ・部品交換・調整重量・スタート展示・水面気象）

対象レースは results テーブルにある（＝実際に行われた）レース。取り込み済みのページは飛ばすので、
途中で落ちても同じコマンドを打ち直せば続きから再開できる。

注意: オッズと直前情報はレース当日にしか出ない。前夜に買い目を決める運用では、
      選定の入力には使えない（過去の市場の歪みを調べる用途・当日運用に変えるときの材料）。

使い方:
  python scripts/scrape_web.py --from 20251101 --to 20261031
  python scripts/scrape_web.py --date 20261101 --pages odds3t,beforeinfo --workers 4
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from db import connect, upsert

BASE = "https://www.boatrace.jp/owpc/pc/race/"
PAGES = ["odds3t", "odds2tf", "oddstf", "beforeinfo"]
UA = {"User-Agent": "boatrace-lab/0.1 (personal research)"}

_ODDS = re.compile(r'class="oddsPoint[^"]*">([^<]*)<')
_TAG = re.compile(r"<[^>]+>")


def _f(s: str):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", s).replace("&nbsp;", " ")).strip()


def fetch(page: str, hd: str, jcd: str, rno: int, session: requests.Session) -> str | None:
    url = f"{BASE}{page}?rno={rno}&jcd={jcd}&hd={hd}"
    for attempt in range(4):
        try:
            r = session.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"fetch failed: {url}")


# ---------------------------------------------------------------- parsers
# 並び順は 2026-11-01 大村1R の払戻（3連単 1-5-2=7320円、2連単 1-5=1910円、2連複 1-5=1450円、
# 単勝 1=150円、複勝 1=170円）と突き合わせて確認済み。

def parse_odds3t(html: str) -> list[tuple]:
    o = _ODDS.findall(html)
    if len(o) != 120:
        return []
    out = []
    for f in range(1, 7):
        seq = [(s, t) for s in range(1, 7) if s != f for t in range(1, 7) if t not in (f, s)]
        for i, (s, t) in enumerate(seq):  # 表は 20 行 × 6 列（列＝1着艇）を行順に並べたもの
            out.append(("sanrensho", f"{f}-{s}-{t}", _f(o[i * 6 + f - 1]), None))
    return out


def parse_odds2tf(html: str) -> list[tuple]:
    o = _ODDS.findall(html)
    if len(o) != 45:
        return []
    out = []
    tan, fuku = o[:30], o[30:]
    for f in range(1, 7):  # 2連単: 5 行 × 6 列（列＝1着艇）
        seconds = [s for s in range(1, 7) if s != f]
        for i, s in enumerate(seconds):
            out.append(("nirensho", f"{f}-{s}", _f(tan[i * 6 + f - 1]), None))
    i = 0
    for s in range(2, 7):  # 2連複: 行＝大きい方の艇、その中で小さい方の艇の昇順
        for f in range(1, s):
            out.append(("nirenfuku", f"{f}-{s}", _f(fuku[i]), None))
            i += 1
    return out


def parse_oddstf(html: str) -> list[tuple]:
    o = _ODDS.findall(html)
    if len(o) != 12:
        return []
    out = [("tansho", str(i + 1), _f(o[i]), None) for i in range(6)]
    for i in range(6):
        lo, _, hi = o[6 + i].partition("-")
        out.append(("fukusho", str(i + 1), _f(lo), _f(hi) if hi else None))
    return out


def parse_beforeinfo(html: str) -> list[dict]:
    m = re.search(r'<table class="is-w748">(.*?)</table>', html, flags=re.S)
    if not m:
        return []
    rows = []
    for body in re.findall(r"<tbody[^>]*>(.*?)</tbody>", m.group(1), flags=re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", body, flags=re.S)
        if len(tds) < 13:
            continue
        lane = _clean(tds[0])
        if not lane.isdigit():
            continue
        parts = [_clean(x) for x in re.findall(r"<li[^>]*>(.*?)</li>", tds[7], flags=re.S)]
        rows.append({
            "lane": int(lane),
            "racer_no": (re.search(r"toban=(\d+)", tds[1]) or [None, None])[1],
            "weight": _f(_clean(tds[3]).replace("kg", "")),
            "exhibition_time": _f(_clean(tds[4])),
            "tilt": _f(_clean(tds[5])),
            "propeller": _clean(tds[6]) or None,
            "parts_changed": [p for p in parts if p],
            "prev_course": _clean(tds[11]) or None,
            "adjust_weight": _f(_clean(tds[12])),
            "prev_st": _clean(tds[14]) if len(tds) > 14 else None,
            "prev_position": _clean(tds[16]) if len(tds) > 16 else None,
        })
    if not rows:
        return []
    # スタート展示: 行がコース順。艇番と ST（"F.01" はフライング）
    start = {}
    for course, (lane, st) in enumerate(re.findall(
            r'table1_boatImage1Number[^>]*>(\d)</span>.*?table1_boatImage1Time[^>]*>([^<]*)<',
            html, flags=re.S), start=1):
        st = st.strip()
        start[int(lane)] = (course, st)
    # 水面気象
    weather = {}
    w = re.search(r'<div class="weather1">(.*?)weather1_stand', html, flags=re.S)
    if w:
        wb = w.group(1)
        for title, data in re.findall(
                r'weather1_bodyUnitLabelTitle">([^<]*)</span>\s*<span class="weather1_bodyUnitLabelData">([^<]*)<', wb):
            weather[title.strip()] = data.strip()
        wt = re.search(r'is-weather.*?weather1_bodyUnitLabelTitle">([^<]*)<', wb, flags=re.S)
        wd = re.search(r'is-wind(\d+)"', wb)
        tm = re.search(r'weather1_title">[^<]*?(\d{1,2}:\d{2})', wb)
        weather = {
            "weather": wt.group(1).strip() if wt else None,
            "air_temp": _f(weather.get("気温", "").replace("℃", "")),
            "wind_speed": _f(weather.get("風速", "").replace("m", "")),
            "wind_direction": int(wd.group(1)) if wd else None,
            "water_temp": _f(weather.get("水温", "").replace("℃", "")),
            "wave_height": _f(weather.get("波高", "").replace("cm", "")),
            "weather_time": tm.group(1) if tm else None,
        }
    for r in rows:
        course, st = start.get(r["lane"], (None, None))
        r["start_course"] = course
        r["start_st_raw"] = st
        r["start_st"] = (-_f(st[1:]) if st.startswith("F") and _f(st[1:]) is not None else _f(st)) if st else None
        r.update(weather)
    return rows


# ---------------------------------------------------------------- main

def work(page, hd, jcd, rno, session):
    html = fetch(page, hd, jcd, rno, session)
    if html is None:
        return page, hd, jcd, rno, []
    if page == "beforeinfo":
        return page, hd, jcd, rno, parse_beforeinfo(html)
    return page, hd, jcd, rno, {"odds3t": parse_odds3t, "odds2tf": parse_odds2tf, "oddstf": parse_oddstf}[page](html)


_DONE_SQL = {
    "odds3t": "select distinct hd, jcd, rno from odds where ticket_type='sanrensho' and hd between ? and ?",
    "odds2tf": "select distinct hd, jcd, rno from odds where ticket_type='nirensho' and hd between ? and ?",
    "oddstf": "select distinct hd, jcd, rno from odds where ticket_type='tansho' and hd between ? and ?",
    "beforeinfo": "select distinct hd, jcd, rno from beforeinfo where hd between ? and ?",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--pages", default=",".join(PAGES))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    d0, d1 = (args.date, args.date) if args.date else (args.date_from, args.date_to)
    if not (d0 and d1):
        ap.error("--date か --from/--to を指定してください")
    pages = args.pages.split(",")

    con = connect()
    races = con.execute(
        "select distinct hd, jcd, rno from results where hd between ? and ? order by hd desc, jcd, rno", (d0, d1)
    ).fetchall()
    jobs = []
    for page in pages:
        done = set(con.execute(_DONE_SQL[page], (d0, d1)).fetchall())
        jobs += [(page, *r) for r in races if tuple(r) not in done]
    print(f"{d0}-{d1}: races {len(races)}, jobs {len(jobs)}", flush=True)

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=args.workers))
    odds_rows, bi_rows, n_done, n_empty, n_err, t0 = [], [], 0, 0, 0, time.time()

    def flush():
        upsert(con, "odds", odds_rows)
        upsert(con, "beforeinfo", bi_rows)
        odds_rows.clear(); bi_rows.clear()

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, *j, session) for j in jobs]
        for fut in as_completed(futs):
            try:
                page, hd, jcd, rno, parsed = fut.result()
            except Exception as e:
                n_err += 1
                print(f"ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue
            n_done += 1
            if not parsed:
                n_empty += 1
            elif page == "beforeinfo":
                for r in parsed:
                    bi_rows.append({
                        "hd": hd, "jcd": jcd, "rno": rno, "lane": r["lane"],
                        "exhibition_time": r["exhibition_time"], "tilt": r["tilt"],
                        "start_course": r["start_course"], "start_st": r["start_st"], "data": r,
                    })
            else:
                for tt, comb, o, omax in parsed:
                    odds_rows.append({"hd": hd, "jcd": jcd, "rno": rno, "ticket_type": tt,
                                      "combination": comb, "odds": o, "odds_max": omax})
            if n_done % 500 == 0:
                flush()
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(jobs)} empty={n_empty} err={n_err} {rate:.1f}/s "
                      f"残り約{(len(jobs) - n_done) / rate / 60:.0f}分", flush=True)
    flush()
    print(f"done: {n_done} pages, empty={n_empty}, err={n_err}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
