"""BOATCAST（公式の映像サービス）のオリジナル展示データを取り込む。場によって 一周／半周ラップ・まわり足・直線 が出る。
  python scripts/scrape_oriten.py --from 20261120 --to 20261123 --workers 6
取得元: https://race.boatcast.jp/txt/{jcd}/bc_oriten_{hd}_{jcd}_{rno:02d}.txt（2025年夏ごろ〜。江戸川など出していない場もある）
展示の後（レース当日）に出る情報なので、前夜の選定には使えない。締切前アラート・当日の検証用。
すでに oriten に入っているレースは取り直さない。取れなかったレースは cache（data/web_cache/oriten_miss.txt）に記録して飛ばす。"""
import argparse, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
MISS = ROOT / "data/web_cache/oriten_miss.txt"
URL = "https://race.boatcast.jp/txt/{j}/bc_oriten_{d}_{j}_{r:02d}.txt"

def num(x):
    try: return float(x)
    except Exception: return None

def fetch(k):
    d, j, r = k
    try:
        t = subprocess.run(["curl", "-sS", "-m", "25", "-A", "Mozilla/5.0", URL.format(j=j, d=d, r=r)], capture_output=True, timeout=40).stdout.decode("utf-8", "replace")
    except Exception:
        return k, None
    if not t.startswith("data="): return k, []
    L = [x.split("\t") for x in t.replace("\r", "").split("\n")[1:] if x.strip()]
    if len(L) < 3: return k, []
    status = L[0][0] if L[0] else ""; heads = [h.replace("　", "").replace(" ", "") for h in L[1]]
    rows = []
    for x in L[2:8]:
        if len(x) < 2 or not x[0].strip().isdigit(): continue
        v = dict(zip(heads, x[2:])); lap = v.get("一周") or v.get("半周ラップ")
        rows.append(dict(hd=d, jcd=j, rno=r, lane=int(x[0]), lap=num(lap), mawari=num(v.get("まわり足")), chokusen=num(v.get("直線")),
                         lap_kind="一周" if "一周" in v else ("半周" if "半周ラップ" in v else None), status=status))
    return k, rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--from", dest="a", required=True); ap.add_argument("--to", dest="b", required=True); ap.add_argument("--workers", type=int, default=6); A = ap.parse_args()
    con = db.connect()
    todo = set((h, j, r) for h, j, r in con.execute("select distinct hd,jcd,rno from entries where hd between ? and ?", (A.a, A.b)))
    done = set((h, j, r) for h, j, r in con.execute("select distinct hd,jcd,rno from oriten where hd between ? and ?", (A.a, A.b)))
    MISS.parent.mkdir(parents=True, exist_ok=True); miss = set(tuple(x.split(",")[:2]) + (int(x.split(",")[2]),) for x in MISS.read_text().split()) if MISS.exists() else set()
    todo = sorted(todo - done - miss); print(f"取得対象 {len(todo)} レース", flush=True)
    n = e = m = 0; buf = []; mf = MISS.open("a")
    with ThreadPoolExecutor(A.workers) as ex:
        for i, (k, rows) in enumerate(ex.map(fetch, todo), 1):
            if rows is None: e += 1
            elif not rows: m += 1; mf.write(f"{k[0]},{k[1]},{k[2]}\n")
            else: buf += rows; n += 1
            if len(buf) >= 3000: db.upsert(con, "oriten", buf); con.commit(); buf = []
            if i % 5000 == 0: print(f"  {i}/{len(todo)} 取得{n} 無し{m} エラー{e}", flush=True)
    if buf: db.upsert(con, "oriten", buf); con.commit()
    mf.close(); print(f"done: 取得 {n} レース、データ無し {m}、エラー {e}", flush=True)
