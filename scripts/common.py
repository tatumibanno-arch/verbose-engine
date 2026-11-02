"""共通ユーティリティ: 設定読み込み・HTTP取得・Supabase接続。"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = ROOT / "config" / "rules.yaml"

BASE_URL = "https://www.boatrace.jp/owpc/pc/race"
HEADERS = {"User-Agent": "boatrace-lab/0.1 (personal research; contact via GitHub)"}
REQUEST_INTERVAL_SEC = 2.0

VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

_last_request_at = 0.0


def load_rules() -> dict:
    with RULES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def polite_get(url: str, params: dict | None = None, retries: int = 2) -> str:
    """公式サイトへ間隔を空けて GET する。失敗時は最大 retries 回リトライ。"""
    global _last_request_at
    for attempt in range(retries + 1):
        wait = REQUEST_INTERVAL_SEC - (time.time() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.time()
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text
        except requests.RequestException as e:
            if attempt == retries:
                raise
            time.sleep(5)
    raise RuntimeError("unreachable")


def supabase_client():
    """環境変数 SUPABASE_URL / SUPABASE_SERVICE_KEY から接続する。"""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定です（環境変数に入れてください）")
    return create_client(url, key)


def today_hd() -> str:
    """JST の今日を YYYYMMDD で返す。"""
    from datetime import datetime, timedelta, timezone

    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y%m%d")
