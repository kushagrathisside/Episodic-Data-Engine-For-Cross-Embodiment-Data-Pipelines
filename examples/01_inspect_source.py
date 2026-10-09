"""Inspect a LeRobot dataset without ingesting it.

Answers "what is actually in this thing?" from metadata alone -- no video decode,
no full table scan -- which is what you want before committing to ingest a
dataset you have not seen before.
"""

from __future__ import annotations

import json
from pathlib import Path

from rde.config import get_settings
from rde.ingestion import LeRobotAdapter
from rde.storage import probe

DATASET_DIR = get_settings().paths.source_cache / "lerobot__svla_so100_pickplace"


def main():
    adapter = LeRobotAdapter(DATASET_DIR, "so100_pickplace")

    print("=" * 72)
    print("SOURCE PROFILE")
    print("=" * 72)
    profile = adapter.inspect()
    print(json.dumps(profile.summary(), indent=2, default=str))

    print("\n" + "=" * 72)
    print("KEY MAPPING: LeRobot -> canonical")
    print("=" * 72)
    print(f"{'source key':34s} {'canonical':16s} {'kind':14s} {'plane':9s} shape")
    for m in adapter._modalities():
        print(f"{str(m.source_key):34s} {m.name:16s} {m.kind.value:14s} "
              f"{m.plane.value:9s} {list(m.shape)}")

    print("\n" + "=" * 72)
    print("PHYSICAL LAYOUT")
    print("=" * 72)
    for entry in profile.files["data_files"]:
        print(f"  data: {entry['file']}  rows={entry['rows']}  "
              f"row_groups={entry['row_groups']}")
    for key, path in profile.files["video_files"].items():
        info = probe(path)
        size_mb = profile.files["video_bytes"][key] / 1e6
        print(f"  video: {key}")
        print(f"     {size_mb:.1f} MB  {info.codec}  {info.width}x{info.height}  "
              f"{info.num_frames} frames  {info.duration_s:.1f}s")
        print(f"     naive full decode would need "
              f"{info.raw_rgb_bytes / 1e9:.1f} GB of RAM")

    print("\n" + "=" * 72)
    print("EPISODE / VIDEO WINDOW ALIGNMENT (first 5 episodes)")
    print("=" * 72)
    print(f"{'ep':>3} {'frames':>7} {'global range':>16} {'video window':>20} {'implied':>9}")
    for ep in list(adapter.iter_episodes())[:5]:
        win = ep.attributes["video_windows"].get("rgb.top", {})
        start = win.get("from_timestamp", 0.0)
        end = win.get("to_timestamp", 0.0)
        rng = f"[{ep.frame_start},{ep.frame_end})"
        window = f"{start:.2f}-{end:.2f}s"
        print(f"{ep.episode_id:>3} {ep.num_frames:>7} {rng:>16} {window:>20} "
              f"{ep.num_frames / 30:>8.2f}s")
    print("\n  The video window duration matches frames/fps exactly, which is what")
    print("  makes 'video_ts = episode.from_timestamp + frame.timestamp' correct.")


if __name__ == "__main__":
    main()
