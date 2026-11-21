"""当日の直前情報・3連単オッズ・今日の結果を1レース分表示する（穴狙いの展開読み用）。curl で1本ずつ取る（1回約9秒）。

    python scripts/analysis/ana_live.py JCD RNO [買い目の型 ...]     例: ana_live.py 12 9 45x 4x5 51x   （x=全。合成オッズを出す）
    python scripts/analysis/ana_live.py results JCD                   今日その場のここまでの結果（決まり手つき）
"""
import subprocess, sys, itertools, re, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import scrape_web as sw  # noqa: E402

hd = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d")


def get(page, jcd, rno=None):
    url = f"{sw.BASE}{page}?jcd={jcd}&hd={hd}" + (f"&rno={rno}" if rno else "")
    r = subprocess.run(["curl", "-sS", "-f", "-m", "60", "--retry", "1", url], capture_output=True)
    return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else ""


if sys.argv[1] == "results":
    h = re.sub(r"<script.*?</script>|<style.*?</style>", " ", get("resultlist", sys.argv[2].zfill(2)), flags=re.S)
    t = sw._clean(h).replace("&yen;", "¥"); i = t.find("レース 3連勝単式")
    print(t[i if i >= 0 else 0:][:3000] if i >= 0 else "今日の結果はまだ無い"); sys.exit()
jcd, rno = sys.argv[1].zfill(2), int(sys.argv[2])
bi = sw.parse_beforeinfo(get("beforeinfo", jcd, rno))
if bi and any(b.get("exhibition_time") for b in bi):
    w = bi[0]
    print(f"天候{w.get('weather')} 風速{w.get('wind_speed')}m 風向コード{w.get('wind_direction')} 波{w.get('wave_height')}cm")
    for b in bi:
        print(f"  {b['lane']}号艇 展示{b.get('exhibition_time')} チルト{b.get('tilt')} スタ展{b.get('start_course')}コース ST{b.get('start_st_raw')} 部品{b.get('parts_changed')} 体重{b.get('weight')}")
else:
    print("直前情報（展示）はまだ出ていない")
od = sw.parse_odds3t(get("odds3t", jcd, rno))
q = {cb.replace("-", ""): 1 / o for _, cb, o, _ in od if o}
if len(q) < 60:
    print("3連単オッズはまだ出ていない"); sys.exit()
S = sum(q.values())
print("頭シェア: " + " ".join(f"{i}:{sum(v for k, v in q.items() if k[0]==str(i))/S:.0%}" for i in range(1, 7)))
print("2着まで(上位10): " + " ".join(f"{a}-{b}:{sum(v for k, v in q.items() if k[:2]==f'{a}{b}')/S:.0%}" for a, b in sorted(itertools.permutations(range(1, 7), 2), key=lambda ab: -sum(v for k, v in q.items() if k[:2] == f"{ab[0]}{ab[1]}"))[:10]))
print("人気順(上位12): " + " ".join(f"{k[0]}-{k[1]}-{k[2]}:{1/v:.1f}" for k, v in sorted(q.items(), key=lambda kv: -kv[1])[:12]))
for pat in sys.argv[3:]:
    ks = [k for k in q if all(a == "x" or a == b for a, b in zip(pat, k))]
    if ks: print(f"型 {pat}: {len(ks)}点 合成{1/sum(q[k] for k in ks):.1f}倍  " + " ".join(f"{k[0]}-{k[1]}-{k[2]}:{1/q[k]:.0f}" for k in sorted(ks, key=lambda k: -q[k])[:8]))
