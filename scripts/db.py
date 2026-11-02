"""SQLite と Parquet の橋渡し。

- 正本は data/parquet/*.parquet（年ごと、git 管理）
- data/boatrace.db は作業用。無ければ Parquet から再構築する

使い方:
  python scripts/db.py rebuild     # Parquet → SQLite を作り直す
  python scripts/db.py export      # SQLite → Parquet（年ごとに書き出す）
  python scripts/db.py stats       # 件数と期間を表示
  python scripts/db.py sql "select ..."   # その場で集計
"""
from __future__ import annotations

import json
import sqlite3
import sys

import pandas as pd

from common import DB_PATH, PARQUET_DIR

TABLES = {
    # 出走表（番組表）: 1艇1行。data に元レコードの全項目を JSON で保持
    "entries": """
        create table if not exists entries (
          hd text not null, jcd text not null, rno int not null, lane int not null,
          racer_no text, racer_name text, grade text, motor_no int, boat_no int,
          close_time text, race_name text,
          data text,
          primary key (hd, jcd, rno, lane)
        )""",
    # 結果: 1艇1行（着順・展示タイム等は data に）
    "results": """
        create table if not exists results (
          hd text not null, jcd text not null, rno int not null, lane int not null,
          position int, data text,
          primary key (hd, jcd, rno, lane)
        )""",
    # レース単位の情報（成績ファイル由来）: 天候・風向・風速・波高など
    "races": """
        create table if not exists races (
          hd text not null, jcd text not null, rno int not null,
          race_name text, distance int, weather text, wind_direction int,
          wind_speed real, wave_height real, data text,
          primary key (hd, jcd, rno)
        )""",
    # 払戻: 1券種1行
    "payouts": """
        create table if not exists payouts (
          hd text not null, jcd text not null, rno int not null,
          ticket_type text not null, combination text not null, payout int,
          primary key (hd, jcd, rno, ticket_type, combination)
        )""",
    # 選定（買い目）: 1買い目1行。実購入は人がやるので paper=1 は検証用、0 は実際に買った分
    "picks": """
        create table if not exists picks (
          hd text not null, jcd text not null, rno int not null,
          ticket_type text not null, combination text not null, amount int not null,
          rules_version int, reason text, paper int default 1,
          primary key (hd, jcd, rno, ticket_type, combination)
        )""",
    # 締切時オッズ（boatrace.jp）: 1券種1組み合わせ1行。複勝は odds=下限, odds_max=上限
    "odds": """
        create table if not exists odds (
          hd text not null, jcd text not null, rno int not null,
          ticket_type text not null, combination text not null, odds real, odds_max real,
          primary key (hd, jcd, rno, ticket_type, combination)
        )""",
    # 直前情報（boatrace.jp）: 1艇1行。展示タイム・チルト・部品交換・スタート展示・水面気象は data に
    "beforeinfo": """
        create table if not exists beforeinfo (
          hd text not null, jcd text not null, rno int not null, lane int not null,
          exhibition_time real, tilt real, start_course int, start_st real,
          data text,
          primary key (hd, jcd, rno, lane)
        )""",
    # 選手期別成績（公式 fan ファイル）: 1選手1期1行。hd は算出期間の最終日。
    # レース日 hd より前に算出期間が終わった最新の行を使えば、未来の情報は混ざらない。
    "racer_stats": """
        create table if not exists racer_stats (
          hd text not null, racer_no text not null,
          year int, term int, period_from text, grade text,
          win_rate real, top2_rate real, avg_st real,
          data text,
          primary key (hd, racer_no)
        )""",
    # 日次検証
    "verifications": """
        create table if not exists verifications (
          hd text primary key, rules_version int,
          races int, bets int, hits int, bet_total int, payout_total int,
          hit_rate real, roi real, note text
        )""",
    # 日次検証のルール別（picks.reason の先頭のルール名ごと）。verifications は全ルールの合計
    "verifications_by_rule": """
        create table if not exists verifications_by_rule (
          hd text not null, rule text not null, rules_version int,
          races int, bets int, hits int, refunds int, bet_total int, payout_total int,
          hit_rate real, roi real,
          primary key (hd, rule)
        )""",
}

KEY_COLS = {
    "entries": ["hd", "jcd", "rno", "lane"],
    "results": ["hd", "jcd", "rno", "lane"],
    "races": ["hd", "jcd", "rno"],
    "payouts": ["hd", "jcd", "rno", "ticket_type", "combination"],
    "picks": ["hd", "jcd", "rno", "ticket_type", "combination"],
    "verifications": ["hd"],
    "verifications_by_rule": ["hd", "rule"],
    "odds": ["hd", "jcd", "rno", "ticket_type", "combination"],
    "beforeinfo": ["hd", "jcd", "rno", "lane"],
    "racer_stats": ["hd", "racer_no"],
}


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    fresh = not DB_PATH.exists()
    # 並列取り込み（月単位×12本＋スクレイプ）で書き込みが重なるので、待ち時間を長めにとり WAL にする
    con = sqlite3.connect(DB_PATH, timeout=600)
    try:
        con.execute("pragma journal_mode=wal")
    except sqlite3.OperationalError:
        pass
    for ddl in TABLES.values():
        con.execute(ddl)
    if fresh:
        _import_parquet(con)
    return con


def upsert(con: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = list(rows[0].keys())
    sql = (
        f"insert or replace into {table} ({','.join(cols)}) "
        f"values ({','.join('?' for _ in cols)})"
    )
    con.executemany(sql, [[_jsonable(r.get(c)) for c in cols] for r in rows])
    con.commit()
    return len(rows)


def _jsonable(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


def _import_parquet(con: sqlite3.Connection) -> None:
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        table = p.stem.rsplit("_", 1)[0]
        if table in TABLES:
            df = pd.read_parquet(p)
            df.to_sql(table, con, if_exists="append", index=False)
    con.commit()


def export_parquet(con: sqlite3.Connection) -> list[str]:
    """年ごとに Parquet を書き出す（既存ファイルは上書き）。"""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for table in TABLES:
        df = pd.read_sql(f"select * from {table}", con)
        if df.empty:
            continue
        for year, g in df.groupby(df["hd"].str[:4]):
            path = PARQUET_DIR / f"{table}_{year}.parquet"
            g.sort_values(KEY_COLS[table]).to_parquet(path, index=False, compression="zstd")
            written.append(path.name)
    return written


def stats(con: sqlite3.Connection) -> dict:
    out = {}
    for table in TABLES:
        row = con.execute(f"select count(*), min(hd), max(hd) from {table}").fetchone()
        out[table] = {"rows": row[0], "from": row[1], "to": row[2]}
    return out


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "rebuild":
        if DB_PATH.exists():
            DB_PATH.unlink()
        con = connect()
        print(json.dumps(stats(con), ensure_ascii=False, indent=2))
    elif cmd == "export":
        con = connect()
        print("\n".join(export_parquet(con)) or "(nothing)")
    elif cmd == "stats":
        print(json.dumps(stats(connect()), ensure_ascii=False, indent=2))
    elif cmd == "sql":
        df = pd.read_sql(sys.argv[2], connect())
        print(df.to_string(index=False, max_rows=200))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
