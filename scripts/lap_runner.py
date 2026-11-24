"""締切前の記録と3連単1点の知らせ（t3-lap-one-v1）を、このクラウド環境で1日分回す。
GitHub Actions からは 2026-11-24 朝に公式サイトの取得が失敗した（全部の再試行で失敗）ため、こちらで動かす。
    python scripts/lap_runner.py --until 21:40（run_in_background で。110分で区切って終わる。終わったら次を起動する）
live_alert.py の run を、枠を [今, until] にして1回呼ぶ。終わったら（または30分ごとに）data/live/log をコミットしてプッシュする。"""
import argparse, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_alert as la  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
BR = "claude/tender-newton-ymc8j0"


WORK = ROOT / "data" / "live" / "log_work"   # 書き込み中のログ（git の外）。30分ごとに data/live/log へ写してコミットする（作業ツリーが途中で汚れないように）


def push_logs(msg: str) -> None:
    import shutil
    sh = lambda *a: subprocess.run(list(a), cwd=ROOT, capture_output=True, text=True)
    for f in WORK.glob("*.csv"):
        shutil.copy2(f, ROOT / "data" / "live" / "log" / f.name)
    sh("git", "add", "data/live/log")
    if sh("git", "diff", "--cached", "--quiet").returncode == 0:
        return
    sh("git", "commit", "-q", "-m", f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01FtFpn2uWpYEZUa2NQTZBzW")
    for i in range(4):
        if sh("git", "pull", "-q", "--rebase", "--autostash", "origin", BR).returncode == 0 and sh("git", "push", "-q", "origin", f"HEAD:{BR}").returncode == 0:
            return
        time.sleep(5 * (i + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--until", default="21:40"); ap.add_argument("--max-minutes", type=int, default=110); a = ap.parse_args()
    import shutil
    WORK.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "data" / "live" / "log").glob(f"{la.to_hd(la.today())}_*.csv"):   # 今日の分を作業場所へ（途中から再開できるように）
        if not (WORK / f.name).exists():
            shutil.copy2(f, WORK / f.name)
    la.LOG_DIR = WORK
    now = la.now_jst().strftime("%H:%M")
    from datetime import timedelta
    cap = (la.now_jst() + timedelta(minutes=a.max_minutes)).strftime("%H:%M")
    a.until = min(a.until, cap)  # この環境のバックグラウンド実行は2時間で切れるので、その前に区切って終わる（終わったら司令塔が次を起動）
    orig = la.cfg
    la.cfg = lambda: dict(orig(), slots=[[now, a.until]])
    stop = threading.Event()
    def periodic():
        while not stop.wait(1200):
            push_logs(f"live: 締切前の記録（途中） {la.now_jst():%Y%m%d-%H%M}")
    threading.Thread(target=periodic, daemon=True).start()
    try:
        la.cmd_run(0, False, False)
    finally:
        stop.set(); push_logs(f"live: 締切前の記録 {la.now_jst():%Y%m%d-%H%M}")
