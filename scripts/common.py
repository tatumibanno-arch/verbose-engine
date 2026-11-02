"""共通: パス・設定・日付。外部サービスには依存しない。"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = ROOT / "config" / "rules.yaml"
DATA_DIR = ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"
DB_PATH = DATA_DIR / "boatrace.db"        # gitignore（Parquet から再構築できる）
CACHE_DIR = DATA_DIR / "lzh_cache"        # gitignore（公式ファイルの一時置き場）
NOTES_DIR = ROOT / "notes"

JST = timezone(timedelta(hours=9))

VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}


def load_rules() -> dict:
    with RULES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def today() -> date:
    return datetime.now(JST).date()


def tomorrow() -> date:
    return today() + timedelta(days=1)


def to_hd(d: date) -> str:
    return d.strftime("%Y%m%d")


def from_hd(hd: str) -> date:
    return date(int(hd[:4]), int(hd[4:6]), int(hd[6:]))


def venue_name(jcd: str) -> str:
    return VENUES.get(str(jcd).zfill(2), str(jcd))
