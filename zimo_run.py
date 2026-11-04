# SPDX-License-Identifier: MIT
"""Launcher used by the `zimo` command (and zimo.bat / zimo.sh).

Starts the Streamlit Studio with the project's virtual environment, on a free
port (defaults to 8769, or $ZIMO_PORT).

    zimo                 start on 8769 (next free port if busy)
    zimo --port 8800     start on a specific port
    zimo --no-browser    do not open a browser tab
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def venv_python() -> Path | None:
    candidate = APP_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt"
                                     else "bin/python")
    return candidate if candidate.is_file() else None


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        if port_is_free(port):
            return port
    raise SystemExit(f"[x] no free port in {preferred}..{preferred + 19}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="zimo",
        description="启动 Zimo Studio（本地端侧语言模型工作台）",
    )
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("ZIMO_PORT", 8769)),
                        help="监听端口（默认 8769，被占用时自动 +1 尝试）")
    parser.add_argument("--no-browser", action="store_true",
                        help="不自动打开浏览器")
    parser.add_argument("--address", default="127.0.0.1",
                        help="绑定地址（默认仅本机 127.0.0.1）")
    args = parser.parse_args()

    python = venv_python()
    if python is None:
        # No .venv yet: fall back to the current interpreter if it can import
        # streamlit (handy when someone pip-installed the deps system-wide).
        probe = subprocess.run([sys.executable, "-c", "import streamlit"],
                               capture_output=True)
        if probe.returncode != 0:
            print("[!] 还没安装依赖。先在项目目录运行 setup.bat / ./setup.sh",
                  file=sys.stderr)
            print(f"    项目目录：{APP_DIR}", file=sys.stderr)
            return 1
        python = Path(sys.executable)
    if not (APP_DIR / "app.py").is_file():
        print(f"[x] 找不到 app.py（{APP_DIR}）", file=sys.stderr)
        return 1

    port = pick_port(args.port)
    url = f"http://{args.address}:{port}"
    print(f"Zimo Studio -> {url}")
    print("按 Ctrl+C 停止\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - headless is fine
            pass

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    command = [str(python), "-m", "streamlit", "run", str(APP_DIR / "app.py"),
               "--server.port", str(port),
               "--server.address", args.address,
               "--browser.gatherUsageStats", "false"]
    try:
        return subprocess.call(command, cwd=str(APP_DIR), env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
