#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def start_process(name: str, cmd: list[str], log_dir: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("a", encoding="utf-8")
    print(f"Starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)


def parse_gpu_list(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    parts = [p for p in spec.split(",") if p.strip()]
    ids: list[int] = []
    for part in parts:
        part = part.strip()
        if not part.isdigit():
            raise ValueError(f"Invalid GPU id '{part}' in '{spec}'")
        ids.append(int(part))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate GPU id in '{spec}'")
    return ids


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
    parser = argparse.ArgumentParser(description="Start vLLM serve processes for two nodes.")
    parser.add_argument("--node", choices=["node0", "node1", "both"], default="both")
    parser.add_argument("--model", default="/mnt/shared/models/SWE-agent-LM-32B")
    parser.add_argument("--port0", type=int, default=8100)
    parser.add_argument("--port1", type=int, default=8200)
    parser.add_argument("--tp0", type=int, default=2)
    parser.add_argument("--tp1", type=int, default=2)
    parser.add_argument("--gpus0", default="0,1", help="Comma-separated GPU ids for node0, e.g. 0,1")
    parser.add_argument("--gpus1", default="2,3", help="Comma-separated GPU ids for node1, e.g. 2,3")
    parser.add_argument("--quantization", default="fp8")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    processes: list[subprocess.Popen] = []
    try:
        gpus0 = parse_gpu_list(args.gpus0)
        gpus1 = parse_gpu_list(args.gpus1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if gpus0 and gpus1:
        overlap = set(gpus0) & set(gpus1)
        if overlap:
            print(f"Error: GPU overlap between node0 and node1: {sorted(overlap)}", file=sys.stderr)
            sys.exit(1)

    if gpus0 and len(gpus0) != args.tp0:
        print(
            f"Warning: node0 tp={args.tp0} but gpus0 has {len(gpus0)} ids ({args.gpus0})",
            file=sys.stderr,
        )
    if gpus1 and len(gpus1) != args.tp1:
        print(
            f"Warning: node1 tp={args.tp1} but gpus1 has {len(gpus1)} ids ({args.gpus1})",
            file=sys.stderr,
        )

    if args.node in ("node0", "both"):
        env0 = os.environ.copy()
        if gpus0:
            env0["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus0)
        cmd0 = [
            "vllm",
            "serve",
            args.model,
            "--port",
            str(args.port0),
            "--tensor-parallel-size",
            str(args.tp0),
            "--quantization",
            args.quantization,
            "--kv_cache_dtype",
            args.kv_cache_dtype,
        ]
        processes.append(start_process("vllm_node0", cmd0, log_dir, env=env0))

    if args.node in ("node1", "both"):
        env1 = os.environ.copy()
        if gpus1:
            env1["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus1)
        cmd1 = [
            "vllm",
            "serve",
            args.model,
            "--port",
            str(args.port1),
            "--tensor-parallel-size",
            str(args.tp1),
            "--quantization",
            args.quantization,
            "--kv_cache_dtype",
            args.kv_cache_dtype,
        ]
        processes.append(start_process("vllm_node1", cmd1, log_dir, env=env1))

    if not processes:
        print("No processes started.", file=sys.stderr)
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
                print(f"Process exited with codes: {codes}, stopping others")
                terminate_processes(processes)
                sys.exit(1)
            time.sleep(1)
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
