#!/usr/bin/env python3
"""Run the analyzer as a native macOS window rather than a browser tab.

Starts the Streamlit server as a child process, waits for it to answer, and
displays it in a WKWebView window. Closing the window shuts the server down —
which the browser-based launcher could not do, leaving a server running
indefinitely after you were finished with it.

    .venv/bin/python desktop.py
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import webview

PROJECT = Path(__file__).parent
STARTUP_TIMEOUT_SECONDS = 30


def _free_port() -> int:
    """Ask the OS for an unused port.

    Better than a fixed 8501: a second copy, or anything else already on that
    port, would otherwise silently attach to the wrong server.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _responding(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}", timeout=1) as reply:
            return reply.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _start_server(port: int) -> subprocess.Popen:
    log = PROJECT / "data" / "app.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log, "ab")
    return subprocess.Popen(
        [str(PROJECT / ".venv" / "bin" / "streamlit"), "run", str(PROJECT / "app.py"),
         "--server.headless", "true",
         "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        cwd=PROJECT,
        stdout=handle,
        stderr=handle,
        # Own process group, so shutdown reaches Streamlit's children too.
        start_new_session=True,
    )


def main() -> int:
    port = _free_port()
    server = _start_server(port)

    def shutdown() -> None:
        if server.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    atexit.register(shutdown)

    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _responding(port):
            break
        if server.poll() is not None:
            print(f"Server exited early — see {PROJECT / 'data' / 'app.log'}",
                  file=sys.stderr)
            return 1
        time.sleep(0.3)
    else:
        shutdown()
        print(f"Server did not start within {STARTUP_TIMEOUT_SECONDS}s",
              file=sys.stderr)
        return 1

    webview.create_window(
        "Golf Swing Analyzer",
        f"http://localhost:{port}",
        width=1280,
        height=900,
        min_size=(900, 600),
    )
    webview.start()          # blocks until the window is closed
    shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
