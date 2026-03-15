"""
Preprocess each trainable song: load MP3, extract features (chroma CQT), build frame-level labels.
Writes songs/<index>/features.npy and songs/<index>/labels.npy. Skips if both already exist.
"""
import json
import os
from pathlib import Path

import numpy as np

# Optional: use librosa for audio + chroma
try:
    import librosa
except ImportError:
    librosa = None

SR = 22050
FPS = 10
HOP_LENGTH = SR // FPS  # 2205
N_FFT = 2048
N_CHROMA = 12

TRAINABLE_PATH = "data/trainable_songs.json"
CORRECTED_PATH = "data/MIR-CE500_corrected.json"
VOCAB_PATH = "data/chord_vocabulary.json"
SKIP_EXISTING = True


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


def process_one(entry, corrected, vocab):
    index = entry["index"]
    index_raw = entry["index_raw"]
    path_mp3 = entry["path_mp3"]
    effective_end_sec = entry["effective_end_sec"]

    out_dir = Path("songs") / index
    feat_path = out_dir / "features.npy"
    label_path = out_dir / "labels.npy"
    if SKIP_EXISTING and feat_path.is_file() and label_path.is_file():
        return None

    if not librosa:
        raise RuntimeError("librosa is required for preprocess_audio. Install with: pip install librosa")

    y, sr = librosa.load(path_mp3, sr=SR, mono=True, duration=effective_end_sec)
    if len(y) == 0:
        return "empty_audio"

    # Chroma CQT @ 10 fps
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        n_chroma=N_CHROMA,
    )
    # (n_chroma, n_frames) -> (n_frames, n_chroma)
    features = chroma.T.astype(np.float32)
    n_frames = features.shape[0]

    frame_times = (np.arange(n_frames) + 0.5) / FPS  # frame center in seconds
    segments = corrected.get(index_raw, [])
    # Clip segment times to effective_end_sec so we don't index past end
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
    with open(TRAINABLE_PATH, "r", encoding="utf-8") as f:
        trainable = json.load(f)
    with open(CORRECTED_PATH, "r", encoding="utf-8") as f:
        corrected = json.load(f)
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab = vocab_data["vocabulary"]

    done = 0
    skipped = 0
    errors = []
    for entry in trainable:
        try:
            result = process_one(entry, corrected, vocab)
            if result is None:
                skipped += 1
            else:
                done += 1
        except Exception as e:
            errors.append((entry["index"], str(e)))
    if errors:
        print("Errors:", errors[:10])
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more.")
    print(f"Preprocessed {done} songs, skipped (existing) {skipped}, errors {len(errors)}.")


if __name__ == "__main__":
    main()
