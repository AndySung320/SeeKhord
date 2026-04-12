"""
Count per-class frame frequencies for reduced_25 (method B).

Same segmentation as ChordDataset (fixed segment_frames chunks over each song),
same full-vocab -> reduction_25 mapping as dataset.py. Does NOT apply padding or
pitch augmentation: only frames in [start:end) are counted; IGNORE_INDEX frames
are excluded (same frames that contribute to loss).

Run from project root:
  python scripts/count_reduced25_train_frames.py --config config.yaml
  python scripts/count_reduced25_train_frames.py --config config.cqt84.example.yaml --data-root /content
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import IGNORE_INDEX


def load_yaml_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("Install PyYAML: pip install pyyaml") from e
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_under_root(root: Path, p: str) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_segments(
    indices: list,
    songs_dir: Path,
    segment_frames: int | None,
) -> list[tuple[str, int, int]]:
    """Mirror ChordDataset segment list (train/val/test indices + label path exists)."""
    segments: list[tuple[str, int, int]] = []
    for index in indices:
        label_path = songs_dir / str(index) / "labels.npy"
        feat_path = songs_dir / str(index) / "features.npy"
        if not label_path.is_file() or not feat_path.is_file():
            continue
        T = int(np.load(label_path, mmap_mode="r").shape[0])
        if segment_frames is None:
            segments.append((str(index), 0, T))
        else:
            for start in range(0, T, segment_frames):
                end = min(start + segment_frames, T)
                segments.append((str(index), start, end))
    return segments


def map_full_to_reduced25(labels_1d: np.ndarray, reduction: np.ndarray) -> np.ndarray:
    """Same logic as ChordDataset.__getitem__ (reduction branch only)."""
    n_classes = len(reduction)
    valid = (labels_1d >= 0) & (labels_1d < n_classes)
    reduced = np.full(labels_1d.shape, IGNORE_INDEX, dtype=np.int64)
    reduced[valid] = reduction[labels_1d[valid].astype(np.int64, copy=False)]
    return reduced


def main() -> None:
    p = argparse.ArgumentParser(
        description="Count reduced_25 label frames per class (segment slicing = ChordDataset; no pad, no aug)."
    )
    p.add_argument("--config", type=Path, default=None, help="YAML (data_root, paths, train.segment_frames)")
    p.add_argument("--data-root", type=Path, default=None, help="Override config data_root")
    p.add_argument("--splits", type=str, default=None, help="Override path to splits.json")
    p.add_argument("--vocabulary", type=str, default=None, help="Override path to chord_vocabulary.json")
    p.add_argument("--songs", type=str, default=None, help="Override songs directory")
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument(
        "--segment-frames",
        type=int,
        default=None,
        help="Chunk length (must match training). Default: config train.segment_frames or 200",
    )
    p.add_argument(
        "--full-song",
        action="store_true",
        help="One segment per song (segment_frames=None), like training with segment_frames unset",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Write counts + fractions to JSON")
    args = p.parse_args()

    cfg: dict = {}
    if args.config is not None and Path(args.config).is_file():
        cfg = load_yaml_config(Path(args.config))

    root = Path(args.data_root if args.data_root is not None else cfg.get("data_root", ".")).resolve()
    paths = cfg.get("paths", {})
    tc = cfg.get("train", {})

    splits_path = Path(
        args.splits if args.splits else str(resolve_under_root(root, paths.get("splits", "data/splits.json")))
    )
    vocab_path = Path(
        args.vocabulary
        if args.vocabulary
        else str(resolve_under_root(root, paths.get("vocabulary", "data/chord_vocabulary.json")))
    )
    songs_dir = Path(
        args.songs if args.songs else str(resolve_under_root(root, paths.get("songs", "songs")))
    )

    if args.full_song:
        segment_frames: int | None = None
    elif args.segment_frames is not None:
        segment_frames = int(args.segment_frames)
    else:
        segment_frames = int(tc.get("segment_frames", 200))

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    indices = splits[args.split]

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    reduction_list = vocab.get("reduction_25")
    if not reduction_list:
        raise SystemExit("chord_vocabulary.json missing reduction_25 (need reduced_25 mapping).")
    reduction = np.asarray(reduction_list, dtype=np.int64)

    segments = build_segments(indices, songs_dir, segment_frames)
    if not segments:
        raise SystemExit(f"No segments found (missing labels.npy/features.npy under {songs_dir}?).")

    counts = np.zeros(25, dtype=np.int64)
    total_slice_frames = 0
    ignored_after_reduce = 0

    for index, start, end in segments:
        labels_mmap = np.load(songs_dir / index / "labels.npy", mmap_mode="r")
        sl = np.asarray(labels_mmap[start:end], dtype=np.int64)
        total_slice_frames += sl.size
        reduced = map_full_to_reduced25(sl, reduction)
        valid = (reduced != IGNORE_INDEX) & (reduced >= 0) & (reduced < 25)
        ignored_after_reduce += int((~valid).sum())
        if valid.any():
            counts += np.bincount(reduced[valid], minlength=25)

    total_valid = int(counts.sum())
    fracs = (counts / max(total_valid, 1)).tolist()

    meta = {
        "data_root": str(root),
        "splits_path": str(splits_path),
        "vocab_path": str(vocab_path),
        "songs_dir": str(songs_dir),
        "split": args.split,
        "segment_frames": segment_frames,
        "method": "B_segment_slices_no_pad_no_augment",
        "num_segments": len(segments),
        "total_slice_frames": total_slice_frames,
        "total_valid_frames": total_valid,
        "ignored_frames_after_reduce": ignored_after_reduce,
    }

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("class_id\tcount\tfraction")
    for c in range(25):
        print(f"{c}\t{int(counts[c])}\t{fracs[c]:.6f}")

    if args.out_json is not None:
        out = {"meta": meta, "counts": counts.tolist(), "fractions": fracs}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
