"""決まり手の取り込み。公式の競走成績ファイル（K）から、レースごとの決まり手と風向（方角）を取り出す。

    python scripts/ingest_kimarite.py --from 20230101 --to 20261113 [--workers 8]

- 取得元: https://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh（data/lzh_cache/K に置き、同じ日は取り直さない）。解凍は 7z。
- 出力: data/parquet/kimarite_{year}.parquet（hd, jcd, rno, kimarite, wind_text, wind_speed, wave）。既にある年のファイルには足して作り直す。
- SQLite（db.py）には入れていない。Parquet を直接読む。
- 決まり手はレース後に分かる情報。予想に使うのは「レース日より前」の分だけ（選手の癖の集計など）。
"""
import argparse, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/lzh_cache/K"; PQ = ROOT / "data/parquet"
URL = "https://www1.mbrace.or.jp/od2/K/{ym}/k{ymd}.lzh"
RACE = re.compile(r"^\s*(\d{1,2})R\s+.*?H\d+m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm")
HEAD = re.compile(r"^\s*着\s+艇\s+登番.*ﾚｰｽﾀｲﾑ\s*(\S*)")


def fetch(hd: str):
    f = CACHE / f"k{hd[2:]}.lzh"
    if not f.exists():
        for i in range(4):
            try:
                r = requests.get(URL.format(ym=hd[:6], ymd=hd[2:]), timeout=30)
                if r.status_code == 404:
                    return hd, None   # 開催なし
                r.raise_for_status(); f.write_bytes(r.content); break
            except Exception:
                time.sleep(2 ** i)
        else:
            return hd, "ERR"
    out = subprocess.run(["7z", "x", "-so", str(f)], capture_output=True)
    if out.returncode != 0:
        return hd, "ERR"
    return hd, out.stdout.decode("cp932", errors="replace")


def parse(hd: str, text: str):
    rows = []; jcd = None; cur = None
    for line in text.splitlines():
        m = re.match(r"^(\d{2})KBGN", line)
        if m:
            jcd = int(m.group(1)); continue
        m = RACE.match(line)
        if m and jcd:
            cur = dict(hd=hd, jcd=jcd, rno=int(m.group(1)), wind_text=m.group(3).replace("　", ""), wind_speed=int(m.group(4)), wave=int(m.group(5)))
            continue
        m = HEAD.match(line)
        if m and cur:
            cur["kimarite"] = m.group(1).replace("　", "").strip() or None
            rows.append(cur); cur = None
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--from", dest="f", required=True); ap.add_argument("--to", dest="t", required=True)
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    d0, d1 = datetime.strptime(a.f, "%Y%m%d"), datetime.strptime(a.t, "%Y%m%d")
    days = [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]
    rows = []; err = []; none = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for hd, text in ex.map(fetch, days):
            if text is None: none += 1
            elif text == "ERR": err.append(hd)
            else: rows += parse(hd, text)
    df = pd.DataFrame(rows)
    print(f"日数 {len(days)}、ファイルなし {none}、エラー {len(err)} {err[:10]}、レース {len(df)}")
    if not len(df): return 1
    for y, g in df.groupby(df.hd.str[:4]):
        p = PQ / f"kimarite_{y}.parquet"
        if p.exists():
            g = pd.concat([pd.read_parquet(p), g]).drop_duplicates(["hd", "jcd", "rno"], keep="last")
        g.sort_values(["hd", "jcd", "rno"]).to_parquet(p, index=False, compression="zstd"); print(p.name, len(g))
    print(df.kimarite.value_counts(dropna=False).to_string())
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
