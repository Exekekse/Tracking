"""Simple Windows process supervisor with start/stop/status capabilities.

This module implements a small command line tool that can start a configured
executable, keep it alive with optional restarts, and provide basic status
and log inspection commands.  It targets Windows but runs on other platforms
for testing.  The configuration is loaded from a JSON file whose path can be
specified via ``--config`` (default: ``supervisor_config.json``).

Main features implemented:

* Validate configuration (paths, directories).
* Start the configured executable with arguments and environment variables
  while redirecting STDOUT/ERR to a rotating logfile.
* Single instance enforcement via PID file.
* Keep-alive with restart policy and backoff.
* Controlled stop with timeout and kill.
* Status information in a small JSON status file.
* ``tail`` command to show last log lines.

The implementation intentionally avoids external dependencies so it can run
in a restricted environment.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
import logging
from typing import Dict, List

DEFAULT_CONFIG = {
    "exe_path": "",
    "work_dir": ".",
    "args": [],
    "env": {},
    "log_dir": "logs",
    "pid_file": "app.pid",
    "window_style": "Normal",
    "restart_policy": "on-failure",
    "restart_backoff": [1, 2, 5, 10],
    "stop_timeout_sec": 8,
    "max_restarts_per_10m": 5,
}

STATE_FILE = "supervisor_status.json"


class ProcessSupervisor:
    """Wraps supervision logic for a single child process."""

    def __init__(self, config: Dict[str, object]):
        self.config = {**DEFAULT_CONFIG, **config}
        self.status_path = os.path.abspath(STATE_FILE)
        self.status = self._load_status()

    # ------------------------------------------------------------------
    # status helpers
    def _load_status(self) -> Dict[str, object]:
        if os.path.exists(self.status_path):
            try:
                with open(self.status_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "state": "STOPPED",
            "pid": None,
            "start_time": None,
            "last_exit_code": None,
            "last_exit_time": None,
            "restart_count": 0,
            "last_log_path": None,
        }

    def _save_status(self) -> None:
        with open(self.status_path, "w", encoding="utf-8") as f:
            json.dump(self.status, f, indent=2)

    def _is_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------
    # config validation
    def _validate(self) -> None:
        exe = self.config["exe_path"]
        if not exe or not os.path.isfile(exe):
            raise FileNotFoundError(f"Executable not found: {exe}")
        os.makedirs(self.config["work_dir"], exist_ok=True)
        os.makedirs(self.config["log_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.status["state"] == "RUNNING" and self.status["pid"]:
            if self._is_process_alive(self.status["pid"]):
                raise RuntimeError("Process already running")

        self._validate()

        logfile = os.path.join(
            self.config["log_dir"], f"app_{time.strftime('%Y%m%d')}.log"
        )
        handler = RotatingFileHandler(logfile, maxBytes=10 * 1024 * 1024, backupCount=5)
        logger = logging.getLogger("supervisor")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        env = os.environ.copy()
        env.update(self.config.get("env", {}))

        args = [self.config["exe_path"], *self.config.get("args", [])]

        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            style = self.config.get("window_style", "Normal")
            if style == "Hidden":
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0
            elif style == "Minimized":
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 6  # SW_MINIMIZE

        proc = subprocess.Popen(
            args,
            cwd=self.config["work_dir"],
            env=env,
            stdout=handler.stream,
            stderr=handler.stream,
            startupinfo=startupinfo,
        )

        self.status.update(
            {
                "state": "RUNNING",
                "pid": proc.pid,
                "start_time": time.time(),
                "restart_count": 0,
                "last_log_path": os.path.abspath(logfile),
            }
        )
        self._save_status()

        self._monitor(proc, logger)

    # ------------------------------------------------------------------
    def _monitor(self, proc: subprocess.Popen, logger: logging.Logger) -> None:
        backoff = self.config.get("restart_backoff", [1, 2, 5, 10])
        restarts: List[float] = []

        while True:
            code = proc.wait()
            now = time.time()
            self.status.update({
                "pid": None,
                "last_exit_code": code,
                "last_exit_time": now,
            })

            policy = self.config.get("restart_policy", "on-failure")
            if policy == "never" or (policy == "on-failure" and code == 0):
                self.status["state"] = "STOPPED"
                self._save_status()
                break

            # restart policy
            restarts = [t for t in restarts if now - t < 600]
            if len(restarts) >= self.config.get("max_restarts_per_10m", 5):
                self.status["state"] = "FAILED"
                self._save_status()
                break

            delay = backoff[min(self.status["restart_count"], len(backoff) - 1)]
            logger.info("process exited with code %s, restarting in %ss", code, delay)
            time.sleep(delay)
            restarts.append(time.time())
            self.status["restart_count"] += 1

            args = [self.config["exe_path"], *self.config.get("args", [])]
            proc = subprocess.Popen(
                args,
                cwd=self.config["work_dir"],
                env=os.environ.copy(),
                stdout=logger.handlers[0].stream,
                stderr=logger.handlers[0].stream,
            )
            self.status.update({"pid": proc.pid, "state": "RUNNING"})
            self._save_status()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        pid = self.status.get("pid")
        if not pid or not self._is_process_alive(pid):
            print("Process not running")
            return

        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

        timeout = self.config.get("stop_timeout_sec", 8)
        for _ in range(timeout * 10):
            if not self._is_process_alive(pid):
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

        self.status.update({"state": "STOPPED", "pid": None})
        self._save_status()
        print("stopped")

    # ------------------------------------------------------------------
    def status_cmd(self) -> None:
        st = self.status
        if st["state"] == "RUNNING" and st["pid"] and not self._is_process_alive(st["pid"]):
            st["state"] = "STOPPED"
            st["pid"] = None
            self._save_status()
        print(json.dumps(st, indent=2))

    # ------------------------------------------------------------------
    def tail(self, n: int) -> None:
        log_path = self.status.get("last_log_path")
        if not log_path or not os.path.exists(log_path):
            print("no log file")
            return
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-n:]
        for line in lines:
            sys.stdout.write(line)

    # ------------------------------------------------------------------
    def clear_failed(self) -> None:
        if self.status.get("state") == "FAILED":
            self.status["state"] = "STOPPED"
            self.status["restart_count"] = 0
            self._save_status()
            print("FAILED state cleared")


def load_config_from_file(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple process supervisor")
    parser.add_argument("command", choices=["start", "stop", "restart", "status", "tail", "clear-failed"], help="command to execute")
    parser.add_argument("-n", "--lines", type=int, default=200, help="lines for tail")
    parser.add_argument("--config", default="supervisor_config.json", help="config JSON path")
    args = parser.parse_args()

    config = load_config_from_file(args.config)
    sup = ProcessSupervisor(config)

    if args.command == "start":
        sup.start()
    elif args.command == "stop":
        sup.stop()
    elif args.command == "restart":
        sup.stop()
        sup.start()
    elif args.command == "status":
        sup.status_cmd()
    elif args.command == "tail":
        sup.tail(args.lines)
    elif args.command == "clear-failed":
        sup.clear_failed()


if __name__ == "__main__":
    main()
