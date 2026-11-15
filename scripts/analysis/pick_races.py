"""G1・SG の節の資料（g1_prep.py の出力）から、翌日の「予想しがいのあるレース」を選び、判断材料だけを並べる。

買い目は出さない（予想は運用者が自分でやる）。運用者の観戦・趣味用。
使い方:
  python scripts/analysis/pick_races.py --date 20261116 \
      --dossier "びわこ=notes/reads/biwako-taisho-2026-11.md" \
      --dossier "丸亀=notes/reads/marugame-kyogokusho-2026-11.md" \
      --out notes/reads/picks-2026-11-16.md [--n 5]

選び方（目安。当たりを約束するものではない）:
  「1号艇の堅さ」＝1号艇の選手の1コース1着率に、モーターと節間の「残り」（マイナスが良い）を足し引きしたもの。
  堅さが上位のレースを「堅そう」、下位のレースを「荒れそう」として混ぜて選ぶ。ドリーム戦・準優・優勝戦は必ず入れる。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SPECIAL = ("ドリーム", "準優", "優勝", "選抜")


def tables(md: str) -> dict[str, list[dict]]:
    """見出し行（| で始まる）ごとに表を読む。キーは1列目の見出し＋2列目の見出し。"""
    out: dict[str, list[dict]] = {}
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            head = [c.strip() for c in ln.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                cells = [c.strip() for c in lines[j].strip("|").split("|")]
                rows.append(dict(zip(head, cells)))
                j += 1
            out.setdefault(f"{head[0]}|{head[1]}", rows)
            i = j
        else:
            i += 1
    return out


def num(s, default=None):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s or "")
    return float(m.group()) if m else default


def course_cell(s: str):
    """'62走 71%/89% ST0.12' → (走, 1着率, 3連率, ST)"""
    m = re.match(r"(\d+)走 (\d+)%/(\d+)% ST([\d.]+)", s or "")
    return tuple(float(x) for x in m.groups()) if m else None


def load(path: str) -> dict:
    md = Path(path).read_text(encoding="utf-8")
    t = tables(md)
    motor = {r["機"]: r for r in t.get("順|機", [])}
    racer = {r["機"]: r for r in t.get("選手|級", [])}
    setsu = {r["機"]: r for r in t.get("選手|機", [])}
    prog = t.get("R|締切", [])
    m = re.search(r"## 番組（(\d+)/(\d+)）", md)
    pday = f"{int(m.group(1)):02d}{int(m.group(2)):02d}" if m else None
    m = re.search(r"\| 直近1年・全レース \| \d+ \| (.+) \|", md)
    venue_course = m.group(1) if m else ""
    days = re.findall(r"^- (\d+/\d+): \d+（1コース (\d+)/(\d+)）", md, re.M)
    return dict(motor=motor, racer=racer, setsu=setsu, prog=prog, pday=pday, venue_course=venue_course, days=days)


def boat(d: dict, lane: int, cell: str) -> dict:
    m = re.match(r"(.+)\((\d+)\)", cell)
    name, mno = (m.group(1).strip(), m.group(2)) if m else (cell, "")
    mo, ra, se = d["motor"].get(mno, {}), d["racer"].get(mno, {}), d["setsu"].get(mno, {})
    return dict(lane=lane, name=name, mno=mno, mo=mo, ra=ra, se=se)


def kata(b1: dict) -> float:
    c = course_cell(b1["ra"].get("1コース"))
    base = c[1] if c else 55.0
    mres = num(b1["mo"].get("残り"), 0.0)
    sres = num(b1["se"].get("残り"), 0.0)
    return base - 15 * mres - 8 * sres


def boat_lines(b: dict) -> list[str]:
    ra, mo, se = b["ra"], b["mo"], b["se"]
    c = ra.get(f"{b['lane']}コース", "-")
    seg = [f"**{b['lane']}号艇 {b['name']}**（{ra.get('級', '-')}）"]
    seg.append(f"{b['lane']}コース成績（直近1年 走 1着/3連 ST）: {c}")
    seg.append(f"直近90日: {ra.get('直近90日: 走・1着率・3連率・ST', '-')}／当地3年: {ra.get('当地3年: 走・1着率・3連率', '-')}／前づけ率 {ra.get('前づけ率', '-')}")
    seg.append(
        f"モーター{b['mno']}号機: 公式2連率 {mo.get('公式2連率', '-')}・{mo.get('走', '-')}走・3連率 {mo.get('3連率', '-')}・残り {mo.get('残り', '-')}・前検 {mo.get('前検', '-')}"
    )
    if se:
        seg.append(
            f"節間: 着順 {se.get('着順（日ごと）', '-')}・残り {se.get('残り', '-')}・平均展示 {se.get('平均展示', '-')}（順位 {se.get('展示順位', '-')}）・平均ST {se.get('平均ST', '-')}・進入 {se.get('進入', '-')}"
        )
    else:
        seg.append("節間: まだ走っていない")
    return ["- " + seg[0], *["  - " + s for s in seg[1:]]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--dossier", action="append", required=True, help="場名=資料のパス")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    races = []
    venues = {}
    for spec in a.dossier:
        vname, path = spec.split("=", 1)
        d = load(path)
        if d["pday"] != a.date[4:]:
            print(f"skip {vname}: 番組が {d['pday']}（{a.date} ではない）")
            continue
        venues[vname] = d
        for r in d["prog"]:
            bs = [boat(d, i, r[str(i)]) for i in range(1, 7)]
            races.append(dict(v=vname, d=d, rno=int(r["R"]), close=r["締切"], kind=r["種別"], boats=bs, kata=kata(bs[0])))

    special = [r for r in races if any(k in r["kind"] for k in SPECIAL)]
    rest = sorted([r for r in races if r not in special], key=lambda r: r["kata"])
    n_left = max(a.n - len(special), 0)
    n_are = (n_left + 1) // 2
    n_kata = n_left - n_are
    are = rest[:n_are]
    katai = rest[::-1][:n_kata]
    for r in special:
        r["why"] = f"{r['kind']}（1号艇の堅さ {r['kata']:.0f}）"
    for r in are:
        b1 = r["boats"][0]
        r["why"] = f"荒れそう: 1号艇 {b1['name']} の堅さ {r['kata']:.0f}（このレース群で下位）。外に攻め手がいるか見どころ"
    for r in katai:
        b1 = r["boats"][0]
        r["why"] = f"堅そう: 1号艇 {b1['name']} の堅さ {r['kata']:.0f}（上位）。2・3着の組み立てが勝負"
    picked = sorted(special + are + katai, key=lambda r: (r["close"], r["v"]))

    hd = f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}"
    L = [f"# 厳選レース {hd}", ""]
    L.append("運用者が自分で予想するための判断材料。買い目は出さない。`scripts/analysis/pick_races.py` で作成（材料は前夜までに分かるものだけ。展示・オッズは当日に見る）。")
    L.append("「堅さ」＝1号艇の選手の1コース1着率（直近1年、%）に、モーターと節間の『残り』（着順を枠と勝率で説明した残り、マイナスが良い）を足し引きした目安。数走の数字なので断定はしない。")
    L.append("風の予報は取り込んでいない（当日の直前情報で見る）。")
    L.append("")
    for vname, d in venues.items():
        L.append(f"- {vname}: 場のコース別1着率（直近1年）{d['venue_course']}。節間の1コース1着: " + ("、".join(f"{x[0]} {x[1]}/{x[2]}" for x in d["days"]) or "まだ無し"))
    L.append("")
    L.append("| 場 | R | 締切 | 種別 | 選んだ理由 |")
    L.append("|---|---|---|---|---|")
    for r in picked:
        L.append(f"| {r['v']} | {r['rno']} | {r['close']} | {r['kind']} | {r['why']} |")
    for r in picked:
        L += ["", f"## {r['v']} {r['rno']}R（締切 {r['close']}、{r['kind']}）", "", f"選んだ理由: {r['why']}", ""]
        fronts = [b for b in r["boats"] if num(b["ra"].get("前づけ率"), 0) >= 15 and b["lane"] >= 3]
        L.append("進入の見込み: " + ("枠なり想定。前づけ率が高い選手: " + "、".join(f"{b['lane']}号艇 {b['name']}（{b['ra'].get('前づけ率')}）" for b in fronts) if fronts else "枠なり想定（前づけ率15%以上の選手なし）"))
        L.append("")
        for b in r["boats"]:
            L += boat_lines(b)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(picked)} races")
    for r in picked:
        print(r["v"], r["rno"], r["kind"], round(r["kata"]), r["why"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
