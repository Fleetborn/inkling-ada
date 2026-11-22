#!/usr/bin/env python3
"""Start an already prepared Inkling Ada build and verify a local reply."""

import argparse
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener


HOST = "127.0.0.1"
MODEL_ID = "inkling-2-mini"
HTTP = build_opener(ProxyHandler({}))


def start_time(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def owns_listener(pid, port):
    proc = Path(f"/proc/{pid}")
    try:
        with (proc / "net/tcp").open() as table:
            next(table)
            sockets = {
                f"socket:[{fields[9]}]"
                for line in table
                if (fields := line.split())[1] == f"0100007F:{port:04X}"
                and fields[3] == "0A"
            }
        for fd in (proc / "fd").iterdir():
            try:
                if os.readlink(fd) in sockets:
                    return True
            except FileNotFoundError:
                continue
    except (OSError, StopIteration):
        return False
    return False


def request_json(url, deadline, body=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Startup deadline expired")
    request = Request(url)
    if body is not None:
        request = Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    if body is None:
        remaining = min(remaining, 2)
    with HTTP.open(request, timeout=remaining) as response:
        return json.load(response)


def verify(endpoint, port, pid, started, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if start_time(pid) != started:
            raise RuntimeError("Inkling Ada exited before verification")
        if owns_listener(pid, port):
            try:
                models = request_json(endpoint + "/models", deadline)["data"]
                if any(model["id"] == MODEL_ID for model in models):
                    break
            except (OSError, ValueError, KeyError, TypeError):
                pass
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    else:
        raise RuntimeError(f"Inkling Ada did not become ready within {timeout}s")

    reply = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Reply with exactly READY."}],
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": 512,
        "stream": False,
    }
    try:
        choice = request_json(endpoint + "/chat/completions", deadline, reply)["choices"][0]
        if choice["finish_reason"] != "stop" or choice["message"]["content"].strip() != "READY":
            raise RuntimeError("Completion did not end normally with exactly READY")
    except (AttributeError, KeyError, IndexError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Local completion failed: {exc}") from exc
    if time.monotonic() > deadline:
        raise RuntimeError(f"Local completion exceeded the {timeout}s startup deadline")
    if start_time(pid) != started or not owns_listener(pid, port):
        raise RuntimeError("Inkling Ada stopped owning the listener during verification")


def stop_new_process(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            continue


def start_or_reuse(args, settings):
    port = args.port
    endpoint = f"http://{HOST}:{port}/v1"
    state_path = args.state_dir / f"server-{port}.json"
    log_path = args.state_dir / f"server-{port}.log"
    try:
        state = json.loads(state_path.read_text())
    except (FileNotFoundError, ValueError):
        state = None
    if isinstance(state, dict):
        pid, started = state.get("pid"), state.get("start_time")
        if type(pid) is int and pid > 0 and started and start_time(pid) == started:
            if state.get("settings") != settings:
                raise RuntimeError(f"Port {port} is reserved by an Inkling Ada launch with different settings")
            verify(endpoint, port, pid, started, args.startup_timeout)
            return {"outcome": "reused", "endpoint": endpoint, "pid": pid, "log": str(log_path)}

    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((HOST, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError(f"Port {port} is in use by another listener. Choose another port.") from exc
            raise

    temp_path = state_path.with_suffix(".tmp")
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            settings["command"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        started = start_time(process.pid)
        if started is None:
            raise RuntimeError("Inkling Ada exited immediately")
        temp_path.write_text(json.dumps({"pid": process.pid, "start_time": started, "settings": settings}) + "\n")
        temp_path.replace(state_path)
        verify(endpoint, port, process.pid, started, args.startup_timeout)
    except BaseException:
        stop_new_process(process)
        temp_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        raise
    return {"outcome": "started", "endpoint": endpoint, "pid": process.pid, "log": str(log_path)}


def interrupted(_signal_number, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Inkling Ada checkout containing serve.sh")
    parser.add_argument("--engine", required=True, type=Path, help="trusted ninfer-serve executable")
    parser.add_argument("--model", required=True, type=Path, help="prepared .ninfer model artifact")
    parser.add_argument("--port", type=int, default=8080)
    state_home = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    parser.add_argument("--state-dir", type=Path, default=state_home / "inkling-ada")
    parser.add_argument("--startup-timeout", type=int, default=90, help="seconds for readiness and a completed reply")
    parser.add_argument("--spec", choices=["none", "dspark"], default="none")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Run this launcher on the target Linux machine")
    if not 1 <= args.port <= 65535 or args.startup_timeout < 1:
        parser.error("Port must be 1–65535 and startup timeout must be positive")
    args.state_dir = args.state_dir.expanduser().resolve()
    launcher, engine, model = (
        path.expanduser().resolve() for path in (args.repo / "serve.sh", args.engine, args.model)
    )
    for path in (launcher, engine, model):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if not os.access(engine, os.X_OK):
        parser.error(f"Engine is not executable: {engine}")

    import fcntl

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    log = args.state_dir / f"server-{args.port}.log"
    try:
        args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (args.state_dir / f"server-{args.port}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another launch on port {args.port} is in progress") from exc
            inputs = []
            for path in (launcher, engine, model):
                stat = path.stat()
                inputs.append([stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns])
            settings = {
                "command": ["bash", str(launcher), str(engine), str(model), str(args.port), args.spec],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "inputs": inputs,
            }
            print(json.dumps(start_or_reuse(args, settings)))
    except KeyboardInterrupt:
        print(f"Launch interrupted. Log: {log}", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{exc}. Log: {log}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
