"""
Build trainable songs list: only songs with both MP3 and chord annotations.
Output: data/trainable_songs.json
If metadata.json is missing (e.g. download didn't save it), duration is read from the MP3 file.
MP3 may live in songs/<index>/ or raw_songs/<index>/ (after scripts/move_mp3_to_raw_songs.py).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SONGS_DIR = ROOT / "songs"
RAW_SONGS_DIR = ROOT / "raw_songs"
CORRECTED_PATH = ROOT / "data/MIR-CE500_corrected.json"
OUTPUT_PATH = ROOT / "data/trainable_songs.json"
# Duration mismatch: if annotation end and audio duration differ by more than this (sec) or ratio, we still include
# but record effective length as min(audio_duration, annotation_end) and store duration_diff_sec.
MAX_DURATION_DIFF_SEC = 5
MAX_DURATION_DIFF_RATIO = 0.05


def get_audio_duration_sec(mp3_path: Path) -> float:
    """Read duration from MP3 file. Returns None if unavailable."""
    try:
        import librosa
        return float(librosa.get_duration(path=str(mp3_path)))
    except Exception:
        return None


def main():
    with open(CORRECTED_PATH, "r", encoding="utf-8") as f:
        corrected = json.load(f)

    trainable = []
    for song_dir in sorted(SONGS_DIR.iterdir()):
        if not song_dir.is_dir():
            continue
        index = song_dir.name
        index_raw = str(int(index)) if index.isdigit() else index

        mp3_path = song_dir / f"{index}.mp3"
        if not mp3_path.is_file():
            alt = RAW_SONGS_DIR / index / f"{index}.mp3"
            if alt.is_file():
                mp3_path = alt
            else:
                continue

        if index_raw not in corrected:
            continue

        # Prefer metadata.json for duration; if missing, read from MP3
        audio_duration_sec = None
        meta_path = song_dir / "metadata.json"
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            audio_duration_sec = meta.get("duration")
            if audio_duration_sec is not None:
                audio_duration_sec = int(audio_duration_sec)
        if audio_duration_sec is None or audio_duration_sec <= 0:
            audio_duration_sec = get_audio_duration_sec(mp3_path)
            if audio_duration_sec is None or audio_duration_sec <= 0:
                continue
        audio_duration_sec = float(audio_duration_sec)

        segments = corrected[index_raw]
        if not segments:
            continue
        last_end = float(segments[-1][1])
        annotation_end_sec = last_end

        duration_diff_sec = abs(audio_duration_sec - annotation_end_sec)
        duration_diff_ratio = duration_diff_sec / max(audio_duration_sec, 1e-6)
        if duration_diff_sec > MAX_DURATION_DIFF_SEC and duration_diff_ratio > MAX_DURATION_DIFF_RATIO:
            effective_end = min(audio_duration_sec, annotation_end_sec)
        else:
            effective_end = min(audio_duration_sec, annotation_end_sec)

        path_mp3 = str(mp3_path.relative_to(ROOT)).replace("\\", "/")
        entry = {
            "index": index,
            "index_raw": index_raw,
            "annotation_end_sec": annotation_end_sec,
            "audio_duration_sec": audio_duration_sec,
            "effective_end_sec": effective_end,
            "path_mp3": path_mp3,
            "duration_diff_sec": round(duration_diff_sec, 4),
        }
        trainable.append(entry)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(trainable, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUTPUT_PATH}: {len(trainable)} trainable songs.")


if __name__ == "__main__":
    main()
