#!/usr/bin/env python3
import argparse
import csv
import json
import random
import statistics
from pathlib import Path

def read_rows(path: Path, context_col: str) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or context_col not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(f"Missing column '{context_col}' in CSV. Available: {available}")
        for row in reader:
            instance_id = (row.get("instance_id") or "").strip()
            if not instance_id:
                continue
            raw_context = (row.get(context_col) or "").strip()
            if not raw_context:
                continue
            try:
                context_tokens = int(raw_context)
            except ValueError:
                continue
            rows.append({"instance_id": instance_id, "context_tokens": context_tokens})
    return rows

def select_rows(rows: list[dict], count: int, sample_method: str, seed: int) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(rows):
        raise ValueError(f"count {count} exceeds available rows {len(rows)}")
    if count == len(rows):
        return list(rows)
    if sample_method == "random":
        rng = random.Random(seed)
        return rng.sample(rows, count)
    if sample_method == "head":
        return list(rows[:count])
    if sample_method == "tail":
        return list(rows[-count:])
    raise ValueError(f"unknown sample method: {sample_method}")

def read_id_list(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids

def write_id_list(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n")

def stats_for_ids(ids: list[str], context_by_id: dict[str, int]) -> dict:
    values = [context_by_id[i] for i in ids if i in context_by_id]
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Create instance splits from a CSV history file.")
    parser.add_argument("--input", default="unbalance1_history_api.csv", help="CSV file to read")
    parser.add_argument(
        "--context-col",
        default="effective_context_tokens",
        help="Column used as context length (default: effective_context_tokens)",
    )
    parser.add_argument("--count", type=int, default=300, help="Number of instances to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sample-method",
        choices=["random", "head", "tail", "extremes"],
        default="random",
        help="How to pick the pool before sorting (long-short mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["long-short", "random"],
        default="long-short",
        help="Split mode",
    )
    parser.add_argument("--source-list", help="Optional ID list for random mode")
    parser.add_argument("--outdir", default="splits", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path, args.context_col)
    context_by_id = {row["instance_id"]: row["context_tokens"] for row in rows}

    if args.mode == "long-short":
        if args.count % 2 != 0:
            raise SystemExit("count must be even for long-short mode")
        if args.count > len(rows):
            raise SystemExit(f"count {args.count} exceeds available rows {len(rows)}")
        half = args.count // 2
        if args.sample_method == "extremes":
            sorted_rows = sorted(rows, key=lambda row: row["context_tokens"])
            short_rows = sorted_rows[:half]
            long_rows = sorted_rows[-half:]
            selected_sorted = short_rows + long_rows
        else:
            selected = select_rows(rows, args.count, args.sample_method, args.seed)
            selected_sorted = sorted(selected, key=lambda row: row["context_tokens"])
            short_rows = selected_sorted[:half]
            long_rows = selected_sorted[-half:]

        short_ids = [row["instance_id"] for row in short_rows]
        long_ids = [row["instance_id"] for row in reversed(long_rows)]
        selected_ids = [row["instance_id"] for row in selected_sorted]

        write_id_list(outdir / "short_ids.txt", short_ids)
        write_id_list(outdir / "long_ids.txt", long_ids)
        write_id_list(outdir / "selected_ids.txt", selected_ids)

        summary = {
            "mode": args.mode,
            "input": str(input_path),
            "context_col": args.context_col,
            "count": args.count,
            "seed": args.seed,
            "sample_method": args.sample_method,
            "short": stats_for_ids(short_ids, context_by_id),
            "long": stats_for_ids(long_ids, context_by_id),
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        return

    if args.source_list:
        source_ids = read_id_list(Path(args.source_list))
    else:
        source_ids = [row["instance_id"] for row in rows]

    if args.count > len(source_ids):
        raise SystemExit(f"count {args.count} exceeds available IDs {len(source_ids)}")

    rng = random.Random(args.seed)
    selected_ids = rng.sample(source_ids, args.count)
    rng.shuffle(selected_ids)
    half = args.count // 2
    node0 = selected_ids[:half]
    node1 = selected_ids[half:]

    write_id_list(outdir / "random_ids.txt", selected_ids)
    write_id_list(outdir / "random_node0.txt", node0)
    write_id_list(outdir / "random_node1.txt", node1)

    summary = {
        "mode": args.mode,
        "input": str(input_path),
        "context_col": args.context_col,
        "count": args.count,
        "seed": args.seed,
        "node0": stats_for_ids(node0, context_by_id),
        "node1": stats_for_ids(node1, context_by_id),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
