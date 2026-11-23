"""競走成績（K）キャッシュから、全艇の着順欄・ST欄の元表記を取り出して Parquet にする。

目的: results.position は数字か NULL で、F/L/欠/失/転 などの区別が消えているため、元テキストから復元する。
入力: data/lzh_cache/YYYY/MM/lzh_performance_YYYYMMDD.cache（boatrace-lzh のキャッシュ。pickle、中身は解凍・デコード済みテキスト）
      新規ダウンロードはしない（キャッシュに無い日は報告のみ）。
出力: 1行=1艇。hd,jcd,rno,lane,racer_no,pos_raw,code,st_raw,st

使い方:
  python3 -I scripts/parse_result_codes.py --from 20230501 --to 20261123 --out PATH.parquet
"""
import argparse
import pickle
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "lzh_cache"

K_BLOCK = re.compile(r"^(\d{2})KBGN\s*$(.*?)^\d{2}KEND", flags=re.M | re.S)
RACE_HDR = re.compile(r"^\s*(\d{1,2})R\s")
ROW = re.compile(r"^  (\S{1,2})\s+([1-6])\s+(\d{4})\s")
ST_NUM = re.compile(r"^([FL]?)(\d?\.\d+)$")


def parse_st(raw: str):
    m = ST_NUM.match(raw.strip())
    if not m:
        return float("nan")
    v = float(m.group(2))
    return -v if m.group(1) == "F" else v


def parse_text(hd: str, jcd: str, text: str, odd: list):
    rows = []
    rno = None
    for line in text.split("\n"):
        m = RACE_HDR.match(line)
        if m and "ﾓｰﾀｰ" not in line:
            rno = int(m.group(1))
            continue
        m = ROW.match(line)
        if not m or rno is None:
            continue
        pos_raw = m.group(1)
        st_raw = line[39:47].strip()
        # 想定の固定幅から外れた行（モーター欄が数字でない等）は記録しておく
        if not re.match(r"^\s*\d{1,2}\s", line[22:26]):
            odd.append((hd, jcd, rno, line))
        code = "" if pos_raw.isdigit() else pos_raw
        rows.append({
            "hd": hd, "jcd": jcd, "rno": rno, "lane": int(m.group(2)),
            "racer_no": int(m.group(3)), "pos_raw": pos_raw, "code": code,
            "st_raw": st_raw, "st": parse_st(st_raw),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", default="20230501")
    ap.add_argument("--to", dest="d1", default="20261123")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d0 = date(int(a.d0[:4]), int(a.d0[4:6]), int(a.d0[6:]))
    d1 = date(int(a.d1[:4]), int(a.d1[4:6]), int(a.d1[6:]))

    all_rows, missing, empty, odd = [], [], [], []
    d = d0
    while d <= d1:
        hd = d.strftime("%Y%m%d")
        f = CACHE / f"{d:%Y}" / f"{d:%m}" / f"lzh_performance_{hd}.cache"
        if not f.exists():
            missing.append(hd)
        else:
            files = pickle.load(open(f, "rb")).get("data") or {}
            n0 = len(all_rows)
            for text in files.values():
                for m in K_BLOCK.finditer(text):
                    all_rows += parse_text(hd, m.group(1), m.group(2), odd)
            if len(all_rows) == n0:
                empty.append(hd)
        d += timedelta(days=1)

    df = pd.DataFrame(all_rows)
    df = df.astype({"rno": "int64", "lane": "int64", "racer_no": "int64", "st": "float64"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"rows={len(df)} hd={df.hd.min()}..{df.hd.max()} days={df.hd.nunique()}")
    print("codes:", dict(Counter(df.code).most_common()))
    print(f"cache無し: {len(missing)}日 例={missing[:10]}")
    print(f"キャッシュ有だが行0: {len(empty)}日 例={empty[:10]}")
    print(f"固定幅外の行: {len(odd)} 例={odd[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
