"""LINE Messaging API の push だけを行う小さなモジュール（重い依存なし。GitHub Actions からも使う）。

環境変数:
  LINE_USER_ID        送り先（U から始まるID）。無ければ何もしない
  LINE_CHANNEL_TOKEN  チャネルアクセストークン（長期）。無い場合はネットワークシークレットで
                      api.line.me に Authorization を付ける設定を想定してヘッダー無しで送る
"""
from __future__ import annotations

import os

import requests

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_TEXT_MAX = 4900        # LINE のテキスト1通の上限は5000文字
LINE_MESSAGES_MAX = 5       # push 1回で送れるのは5通まで


def chunks(text: str, size: int = LINE_TEXT_MAX) -> list[str]:
    """行の途中で切らずに size 文字以内に分ける。"""
    out, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + 1 + len(line) > size:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


def send_line(text: str) -> bool:
    """送れたら True。LINE_USER_ID が無ければ False。失敗は例外。"""
    to = os.environ.get("LINE_USER_ID")
    if not to:
        return False
    parts = chunks(text)
    if len(parts) > LINE_MESSAGES_MAX:
        parts = parts[:LINE_MESSAGES_MAX - 1] + ["…長いので残りは notes/ を見てください"]
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(LINE_PUSH_URL, headers=headers, timeout=15,
                      json={"to": to, "messages": [{"type": "text", "text": t} for t in parts]})
    if r.status_code != 200:
        raise RuntimeError(f"LINE push failed: HTTP {r.status_code} {r.text[:200]}")
    return True
