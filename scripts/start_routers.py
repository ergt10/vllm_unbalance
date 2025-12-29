#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def start_router(name: str, backend: str, host: str, port: int, log_dir: Path) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["VLLM_BACKENDS"] = backend
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "tunderreact:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"Starting {name}: {' '.join(cmd)} (VLLM_BACKENDS={backend})")
    return subprocess.Popen(cmd, stdout=log_file, stderr=log_file, cwd=repo_root(), env=env)


def terminate_processes(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    for proc in procs:
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if proc.poll() is None:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start tunderreact routers for two nodes.")
    parser.add_argument("--node", choices=["node0", "node1", "both"], default="both")
    parser.add_argument("--backend0", default="http://localhost:8100")
    parser.add_argument("--backend1", default="http://localhost:8200")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port0", type=int, default=8301)
    parser.add_argument("--port1", type=int, default=8302)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    processes: list[subprocess.Popen] = []

    if args.node in ("node0", "both"):
        processes.append(start_router("router_node0", args.backend0, args.host, args.port0, log_dir))

    if args.node in ("node1", "both"):
        processes.append(start_router("router_node1", args.backend1, args.host, args.port1, log_dir))

    if not processes:
        print("No routers started.", file=sys.stderr)
        sys.exit(1)

    def handle_signal(signum, _frame):
        print(f"Received signal {signum}, stopping...")
        terminate_processes(processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while True:
            exited = [p for p in processes if p.poll() is not None]
            if exited:
                codes = ", ".join(str(p.poll()) for p in exited)
                print(f"Router exited with codes: {codes}, stopping others")
                terminate_processes(processes)
                sys.exit(1)
            time.sleep(1)
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
