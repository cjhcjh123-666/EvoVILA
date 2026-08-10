"""Merge RefCOCO-family image manifests (and optionally a video manifest) into
one training manifest directory consumed by train_s4b.py / eval_s4b.py.

All records are deduplicated by sample_id and written as plain jsonl, so the
merged directory keeps the same schema as each input manifest directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _merge_records(paths: List[Path]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for record in _load_jsonl(path):
            seen[record["sample_id"]] = record
    return list(seen.values())


def _merge_pairs(paths: List[Path]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    seen: set = set()
    for path in paths:
        for pair in _load_jsonl(path):
            key = (pair["left_sample_id"], pair["right_sample_id"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
    return pairs


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    files = ("train.jsonl", "val.jsonl", "train.pairs.jsonl", "val.pairs.jsonl")
    for name in files:
        inputs = [directory / name for directory in args.variant_dirs if (directory / name).exists()]
        if args.video_dir is not None:
            video_path = args.video_dir / name
            if video_path.exists():
                inputs.append(video_path)
        if name.endswith("pairs.jsonl"):
            records = _merge_pairs(inputs)
        else:
            records = _merge_records(inputs)
        _write_jsonl(args.output / name, records)
        print(f"{name}: {len(records)} records")
    print(f"merged manifest written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
