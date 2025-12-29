#!/usr/bin/env python3
import argparse
from pathlib import Path


def read_id_list(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def read_exit_status_instances(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            ids.append(stripped[2:].strip())
    return ids


def dedupe_preserve(ids: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def interleave(
    long_ids: list[str],
    short_ids: list[str],
    start_long: bool,
) -> tuple[list[str], bool]:
    out: list[str] = []
    li = 0
    si = 0
    next_long = start_long
    while li < len(long_ids) or si < len(short_ids):
        if next_long:
            if li < len(long_ids):
                out.append(long_ids[li])
                li += 1
            elif si < len(short_ids):
                out.append(short_ids[si])
                si += 1
            next_long = False
        else:
            if si < len(short_ids):
                out.append(short_ids[si])
                si += 1
            elif li < len(long_ids):
                out.append(long_ids[li])
                li += 1
            next_long = True
    return out, next_long


def write_id_list(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an alternating long/short instance list from exit_statuses and remaining IDs."
    )
    parser.add_argument(
        "--short-exit",
        default="/root/vllm_unbalance/short_node0/exit_statuses_1766988286.353548.yaml",
        help="Exit statuses YAML for short node",
    )
    parser.add_argument(
        "--long-exit",
        default="/root/vllm_unbalance/long_node1/exit_statuses_1766988275.617417.yaml",
        help="Exit statuses YAML for long node",
    )
    parser.add_argument(
        "--short-list",
        default="/root/vllm_unbalance/splits/short_ids.txt",
        help="All short instance IDs (ordered)",
    )
    parser.add_argument(
        "--long-list",
        default="/root/vllm_unbalance/splits/long_ids.txt",
        help="All long instance IDs (ordered)",
    )
    parser.add_argument(
        "--start",
        choices=["long", "short"],
        default="long",
        help="Whether to start with a long or short instance",
    )
    parser.add_argument(
        "--out",
        default="/root/vllm_unbalance/splits/mixed_ids.txt",
        help="Output path for the mixed list",
    )
    args = parser.parse_args()

    short_exit_ids = dedupe_preserve(read_exit_status_instances(Path(args.short_exit)))
    long_exit_ids = dedupe_preserve(read_exit_status_instances(Path(args.long_exit)))

    short_all = read_id_list(Path(args.short_list))
    long_all = read_id_list(Path(args.long_list))

    short_done = set(short_exit_ids)
    long_done = set(long_exit_ids)

    short_remaining = [iid for iid in short_all if iid not in short_done]
    long_remaining = [iid for iid in long_all if iid not in long_done]

    start_long = args.start == "long"
    mixed, next_long = interleave(long_exit_ids, short_exit_ids, start_long)
    remainder, _ = interleave(long_remaining, short_remaining, next_long)
    mixed.extend(remainder)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_id_list(out_path, mixed)

    print(f"Wrote {len(mixed)} instance IDs to {out_path}")
    print(f"Already-run: long={len(long_exit_ids)}, short={len(short_exit_ids)}")
    print(f"Remaining: long={len(long_remaining)}, short={len(short_remaining)}")


if __name__ == "__main__":
    main()
