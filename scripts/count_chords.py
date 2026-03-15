"""
Count chord occurrences from annotations in two ways:
1. With root: per full symbol (e.g. C:maj, D:min, A:7) -> segment count and total duration.
2. Without root: per quality only (e.g. maj, min, 7, N) -> segment count and total duration.
Output: data/chord_counts_with_root.json, data/chord_counts_no_root.json
"""
import json
import os
import re
from collections import defaultdict

CORRECTED_PATH = "data/MIR-CE500_corrected.json"
OUTPUT_WITH_ROOT = "data/chord_counts_with_root.json"
OUTPUT_NO_ROOT = "data/chord_counts_no_root.json"


def chord_to_quality(chord_symbol: str) -> str:
    """Extract quality only (no root). N -> 'N'; Root:quality -> quality; root-only -> 'maj'."""
    s = chord_symbol.strip()
    if s == "N":
        return "N"
    m = re.match(r"^[A-G][#b]?:(.+)$", s, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower() or "maj"
    # Root only (e.g. A, Bb) -> major
    return "maj"


def main():
    with open(CORRECTED_PATH, "r", encoding="utf-8") as f:
        corrected = json.load(f)

    # (segment_count, total_duration_sec)
    with_root = defaultdict(lambda: {"count": 0, "duration_sec": 0.0})
    no_root = defaultdict(lambda: {"count": 0, "duration_sec": 0.0})

    for idx in corrected:
        for seg in corrected[idx]:
            start_s = float(seg[0])
            end_s = float(seg[1])
            chord = seg[2].strip()
            dur = end_s - start_s

            with_root[chord]["count"] += 1
            with_root[chord]["duration_sec"] += dur

            quality = chord_to_quality(chord)
            no_root[quality]["count"] += 1
            no_root[quality]["duration_sec"] += dur

    # Sort and output: with root by chord name, no root by quality then by count desc
    with_root_list = [
        {"chord": c, "count": v["count"], "duration_sec": round(v["duration_sec"], 4)}
        for c, v in sorted(with_root.items(), key=lambda x: -x[1]["count"])
    ]
    no_root_list = [
        {"quality": q, "count": v["count"], "duration_sec": round(v["duration_sec"], 4)}
        for q, v in sorted(no_root.items(), key=lambda x: -x[1]["count"])
    ]

    out = {
        "with_root": with_root_list,
        "summary": {"total_segments": sum(v["count"] for v in with_root.values())},
    }
    os.makedirs(os.path.dirname(OUTPUT_WITH_ROOT), exist_ok=True)
    with open(OUTPUT_WITH_ROOT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out2 = {
        "no_root": no_root_list,
        "summary": {"total_segments": sum(v["count"] for v in no_root.values())},
    }
    with open(OUTPUT_NO_ROOT, "w", encoding="utf-8") as f:
        json.dump(out2, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUTPUT_WITH_ROOT}: {len(with_root_list)} chord types (with root).")
    print(f"Wrote {OUTPUT_NO_ROOT}: {len(no_root_list)} quality types (no root).")
    print("\nTop 10 by count (with root):")
    for row in with_root_list[:10]:
        print(f"  {row['chord']}: count={row['count']}, duration={row['duration_sec']:.1f}s")
    print("\nBy quality (no root):")
    for row in no_root_list:
        print(f"  {row['quality']}: count={row['count']}, duration={row['duration_sec']:.1f}s")


if __name__ == "__main__":
    main()
