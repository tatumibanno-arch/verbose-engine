"""締切前の記録と3連単1点の知らせ（t3-lap-one-v1）を、このクラウド環境で1日分回す。
GitHub Actions からは 2026-11-24 朝に公式サイトの取得が失敗した（全部の再試行で失敗）ため、こちらで動かす。
    nohup python scripts/lap_runner.py --until 21:40 > data/live/runner.log 2>&1 &
live_alert.py の run を、枠を [今, until] にして1回呼ぶ。終わったら（または30分ごとに）data/live/log をコミットしてプッシュする。"""
import argparse, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_alert as la  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
BR = "claude/tender-newton-ymc8j0"


def push_logs(msg: str) -> None:
    sh = lambda *a: subprocess.run(list(a), cwd=ROOT, capture_output=True, text=True)
    sh("git", "add", "data/live/log")
    if sh("git", "diff", "--cached", "--quiet").returncode == 0:
        return
    sh("git", "commit", "-q", "-m", f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01FtFpn2uWpYEZUa2NQTZBzW")
    for i in range(4):
        if sh("git", "pull", "-q", "--rebase", "--autostash", "origin", BR).returncode == 0 and sh("git", "push", "-q", "origin", f"HEAD:{BR}").returncode == 0:
            return
        time.sleep(5 * (i + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--until", default="21:40"); a = ap.parse_args()
    now = la.now_jst().strftime("%H:%M")
    orig = la.cfg
    la.cfg = lambda: dict(orig(), slots=[[now, a.until]])
    stop = threading.Event()
    def periodic():
        while not stop.wait(1800):
            push_logs(f"live: 締切前の記録（途中） {la.now_jst():%Y%m%d-%H%M}")
    threading.Thread(target=periodic, daemon=True).start()
    try:
        la.cmd_run(0, False, False)
    finally:
        stop.set(); push_logs(f"live: 締切前の記録 {la.now_jst():%Y%m%d-%H%M}")
