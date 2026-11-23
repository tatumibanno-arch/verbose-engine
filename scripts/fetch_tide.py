#!/usr/bin/env python3
"""気象庁 潮位表（テキストデータ版）を取得し、場×日×時の Parquet にまとめる。

URL: https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{年}/{地点記号}.txt
書式（1行=1日, 136桁, 固定長）:
  1-72   毎時潮位 3桁x24 (0〜23時, cm)
  73-78  年月日 2桁x3 (年は下2桁, 空白詰め)
  79-80  地点記号
  81-108 満潮 (時刻HHMM 4桁 + 潮位3桁) x4
  109-136 干潮 同上
  欠測/該当なし: 時刻 9999, 潮位 999 -> NaN

使い方: python3 scripts/fetch_tide.py [--workdir DIR] [--out FILE] [--years 2024 2025 2026]
取得は curl（1秒1回、既にあるファイルは取り直さない）。接続拒否時は30秒待って1回だけ再試行し、だめなら終了。
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

BASE = "https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt"
SCRATCH = "/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad"

# 場コード -> (地点記号, 地点名, 理由)
VENUE_STATION = {
    "03": ("TK", "東京", "江戸川(河川,潮影響あり)。最寄りの潮位表地点は東京"),
    "04": ("TK", "東京", "平和島。東京(約8km)。川崎も同距離だが東京を採用"),
    "06": ("MI", "舞阪", "浜名湖。舞阪が至近"),
    "07": ("G5", "形原", "蒲郡。形原が至近(約5km)"),
    "08": ("ZD", "鬼崎", "常滑。鬼崎が至近(約6km)"),
    "09": ("G3", "四日市港", "津。津市に地点なし。四日市港(約28km)、鳥羽(約38km)より近い"),
    "12": ("OS", "大阪", "住之江(淡水プール,参考)。大阪"),
    "13": ("AM", "尼崎", "尼崎(淡水プール,参考)。尼崎が至近"),
    "14": ("KM", "小松島", "鳴門。小松島(約18km)"),
    "15": ("TX", "多度津", "丸亀。多度津が至近(約5km)"),
    "16": ("MM", "水島", "児島。水島(約10km)、宇野(約13km)より近い"),
    "17": ("Q8", "広島", "宮島。広島(約14km)、呉(約22km)より近い"),
    "18": ("QA", "徳山", "徳山。同名地点"),
    "19": ("A1", "弟子待", "下関。関門海峡の弟子待が至近(約2km)。長府は約10km"),
    "20": ("N1", "日明", "若松。日明(約6km)。門司より近い"),
    "22": ("QF", "博多", "福岡。博多"),
    "23": ("KA", "唐津", "唐津(淡水プール,参考)。同名地点"),
    "24": ("NS", "長崎", "大村。大村に地点なし。長崎(約22km)、佐世保(約30km)より近い"),
}


def fetch(year, stn, rawdir):
    path = os.path.join(rawdir, f"{year}_{stn}.txt")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"{BASE}/{year}/{stn}.txt"
    for attempt in (1, 2):
        time.sleep(1)
        r = subprocess.run(
            ["curl", "-sS", "-m", "60", "-A", "Mozilla/5.0", "-o", path, "-w", "%{http_code}", url],
            capture_output=True, text=True)
        code = r.stdout.strip()
        if r.returncode == 0 and code == "200":
            return path
        if os.path.exists(path):
            os.remove(path)
        if attempt == 1:
            print(f"fetch failed {url} code={code} err={r.stderr.strip()}; wait 30s and retry once", file=sys.stderr)
            time.sleep(30)
    print(f"ABORT: cannot fetch {url} (code={code}). {r.stderr.strip()}", file=sys.stderr)
    sys.exit(2)


def hhmm(s):
    s = s.replace(" ", "0")
    if s == "9999":
        return ""
    return s


def lvl(s):
    v = int(s)
    return np.nan if v == 999 else float(v)


def parse(path, year, stn):
    rows = []
    with open(path, encoding="ascii") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            assert len(ln) == 136, (path, len(ln))
            yy, mm, dd = int(ln[72:74]), int(ln[74:76]), int(ln[76:78])
            code = ln[78:80]
            assert code == stn and 2000 + yy == year, (path, ln[72:80])
            hd = f"{2000 + yy:04d}{mm:02d}{dd:02d}"
            hours = []
            for h in range(24):
                v = int(ln[h * 3:h * 3 + 3])
                hours.append(np.nan if v == 999 else float(v))
            hi = [(hhmm(ln[80 + i * 7:84 + i * 7]), lvl(ln[84 + i * 7:87 + i * 7])) for i in range(4)]
            lo = [(hhmm(ln[108 + i * 7:112 + i * 7]), lvl(ln[112 + i * 7:115 + i * 7])) for i in range(4)]
            rows.append((hd, stn, hours, hi, lo))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=f"{SCRATCH}/tide")
    ap.add_argument("--out", default=f"{SCRATCH}/n2/tide.parquet")
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    a = ap.parse_args()
    rawdir = os.path.join(a.workdir, "raw")
    os.makedirs(rawdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    parsed = {}
    for stn in sorted({v[0] for v in VENUE_STATION.values()}):
        for y in a.years:
            p = fetch(y, stn, rawdir)
            parsed[(y, stn)] = parse(p, y, stn)

    out = []
    for jcd, (stn, _, _) in sorted(VENUE_STATION.items()):
        for y in a.years:
            for hd, s, hours, hi, lo in parsed[(y, stn)]:
                lv = [v for v in hours if not np.isnan(v)]
                rng = (max(lv) - min(lv)) if lv else np.nan
                ex = {}
                for i in range(2):
                    ex[f"high{i + 1}_time"], ex[f"high{i + 1}_level"] = hi[i]
                for i in range(2):
                    ex[f"low{i + 1}_time"], ex[f"low{i + 1}_level"] = lo[i]
                for h in range(24):
                    out.append({"jcd": jcd, "hd": hd, "hour": h, "level": hours[h], "station": stn, **ex, "day_range": rng})
    df = pd.DataFrame(out)
    cols = ["jcd", "hd", "hour", "level", "station", "high1_time", "high1_level", "high2_time", "high2_level",
            "low1_time", "low1_level", "low2_time", "low2_level", "day_range"]
    df = df[cols]
    df["hour"] = df["hour"].astype("int8")
    df.to_parquet(a.out, index=False, compression="zstd")
    print(f"wrote {a.out} rows={len(df)}")


if __name__ == "__main__":
    main()
