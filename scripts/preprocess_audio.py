"""
Preprocess each trainable song: load MP3, extract features, build frame-level labels.
Writes <songs_dir>/<index>/features.npy and labels.npy. Skips if both already exist.

Modes:
  chroma — 12-d chroma CQT (default, matches legacy songs/)
  cqt84  — 84-bin CQT magnitude in dB (use a separate songs_dir, e.g. songs_cqt84)

Paths are relative to project root (parent of scripts/).

Parallel: use -j N (N>1) or -j 0 for all logical CPUs. Each worker loads
corrected/vocab once via initializer (not per song).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

ROOT = Path(__file__).resolve().parent.parent

SR = 22050
FPS = 10
HOP_LENGTH = SR // FPS  # 2205
N_CHROMA = 12
N_CQT_BINS = 84
BINS_PER_OCTAVE = 12

TRAINABLE_PATH = ROOT / "data/trainable_songs.json"
CORRECTED_PATH = ROOT / "data/MIR-CE500_corrected.json"
VOCAB_PATH = ROOT / "data/chord_vocabulary.json"

# Filled by _init_worker in child processes (avoids pickling huge dicts per task).
_W: dict = {}


def _init_worker(corrected_path: str, vocab_path: str, songs_dir_str: str, mode: str, skip_existing: bool) -> None:
    global _W
    with open(corrected_path, encoding="utf-8") as f:
        corrected = json.load(f)
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)["vocabulary"]
    _W = {
        "corrected": corrected,
        "vocab": vocab,
        "songs_dir": Path(songs_dir_str),
        "mode": mode,
        "skip_existing": skip_existing,
    }


def _worker_entry(entry: dict):
    """Returns (index, result, error_message). result: None skip, tuple shape ok, 'empty_audio'."""
    idx = entry.get("index", "?")
    try:
        r = process_one(
            entry,
            _W["corrected"],
            _W["vocab"],
            _W["songs_dir"],
            _W["mode"],
            _W["skip_existing"],
        )
        return (idx, r, None)
    except Exception as e:
        return (idx, None, str(e))


def segments_to_frame_labels(segments, frame_times, vocab):
    """Map segment list [start, end, chord] to label per frame. Frame time in seconds."""
    out = np.zeros(len(frame_times), dtype=np.int64)
    for i, t in enumerate(frame_times):
        for start_s, end_s, chord in segments:
            if start_s <= t < end_s:
                out[i] = vocab.get(chord.strip(), vocab.get("N", 0))
                break
        else:
            out[i] = vocab.get("N", 0)
    return out


def extract_features_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=N_CHROMA,
    )
    return chroma.T.astype(np.float32)


def extract_features_cqt84(y: np.ndarray, sr: int) -> np.ndarray:
    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_bins=N_CQT_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    mag = np.abs(cqt)
    ref = float(np.max(mag))
    if ref < 1e-10:
        ref = 1.0
    db = librosa.amplitude_to_db(mag, ref=ref, amin=1e-10)
    return db.T.astype(np.float32)


def process_one(entry, corrected, vocab, songs_dir: Path, mode: str, skip_existing: bool):
    index = entry["index"]
    index_raw = entry["index_raw"]
    path_mp3 = Path(entry["path_mp3"])
    if not path_mp3.is_absolute():
        path_mp3 = ROOT / path_mp3
    effective_end_sec = entry["effective_end_sec"]

    out_dir = songs_dir / index
    feat_path = out_dir / "features.npy"
    label_path = out_dir / "labels.npy"
    if skip_existing and feat_path.is_file() and label_path.is_file():
        return None

    if not librosa:
        raise RuntimeError("librosa is required for preprocess_audio. Install with: pip install librosa")

    y, sr = librosa.load(str(path_mp3), sr=SR, mono=True, duration=effective_end_sec)
    if len(y) == 0:
        return "empty_audio"

    if mode == "chroma":
        features = extract_features_chroma(y, sr)
    elif mode == "cqt84":
        features = extract_features_cqt84(y, sr)
    else:
        raise ValueError(f"unknown mode: {mode}")

    n_frames = features.shape[0]

    frame_times = (np.arange(n_frames) + 0.5) / FPS
    segments = corrected.get(index_raw, [])
    segments_clip = []
    for start_s, end_s, chord in segments:
        start_s, end_s = float(start_s), float(end_s)
        if end_s <= 0 or start_s >= effective_end_sec:
            continue
        start_s = max(0, start_s)
        end_s = min(effective_end_sec, end_s)
        if start_s < end_s:
            segments_clip.append((start_s, end_s, chord))
    labels = segments_to_frame_labels(segments_clip, frame_times.tolist(), vocab)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, features)
    np.save(label_path, labels)
    return features.shape


def main():
    p = argparse.ArgumentParser(description="Preprocess songs to features.npy + labels.npy")
    p.add_argument(
        "--mode",
        choices=("chroma", "cqt84"),
        default="chroma",
        help="chroma: 12-d chroma CQT; cqt84: 84-bin log-magnitude CQT",
    )
    p.add_argument(
        "--songs-dir",
        type=str,
        default="songs",
        help="Output folder under project root (use e.g. songs_cqt84 for cqt84 to avoid overwriting chroma)",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Recompute even if features.npy and labels.npy already exist",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers (1 = sequential). Use 0 for os.cpu_count().",
    )
    args = p.parse_args()

    songs_dir = Path(args.songs_dir)
    if not songs_dir.is_absolute():
        songs_dir = ROOT / songs_dir
    skip_existing = not args.no_skip

    with open(TRAINABLE_PATH, "r", encoding="utf-8") as f:
        trainable = json.load(f)
    with open(CORRECTED_PATH, "r", encoding="utf-8") as f:
        corrected = json.load(f)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab = vocab_data["vocabulary"]

    total = len(trainable)
    n_jobs = args.jobs if args.jobs > 0 else max(1, os.cpu_count() or 1)
    n_jobs = min(n_jobs, max(1, total))
    print(f"Project root: {ROOT}")
    print(f"mode: {args.mode}  songs_dir: {songs_dir}")
    print(f"Trainable songs: {total}  jobs: {n_jobs}")
    done = 0
    skipped = 0
    errors = []

    def _handle_one(idx: str, result, err, i: int) -> None:
        nonlocal done, skipped
        tag = f"[{i}/{total}] {idx}"
        if err:
            errors.append((idx, err))
            print(f"{tag} ERROR: {err}")
            return
        if result is None:
            skipped += 1
            print(f"{tag} skip (already exists)")
        elif result == "empty_audio":
            errors.append((idx, "empty_audio"))
            print(f"{tag} ERROR: empty_audio")
        else:
            done += 1
            print(f"{tag} ok T={result[0]} F={result[1]}")

    if n_jobs == 1:
        for i, entry in enumerate(trainable, start=1):
            idx = entry["index"]
            try:
                result = process_one(entry, corrected, vocab, songs_dir, args.mode, skip_existing)
                _handle_one(idx, result, None, i)
            except Exception as e:
                _handle_one(idx, None, str(e), i)
    else:
        initargs = (
            str(CORRECTED_PATH.resolve()),
            str(VOCAB_PATH.resolve()),
            str(songs_dir.resolve()),
            args.mode,
            skip_existing,
        )
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=initargs) as pool:
            future_map = {pool.submit(_worker_entry, e): e for e in trainable}
            for completed, fut in enumerate(as_completed(future_map), start=1):
                idx, result, err = fut.result()
                _handle_one(idx, result, err, completed)
    if errors:
        print("Errors:", errors[:10])
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more.")
    print(f"Preprocessed {done} songs, skipped (existing) {skipped}, errors {len(errors)}.")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
