import yt_dlp
import json
import os
from typing import Any, Dict
import time
import random

def zero_pad_index(idx: str, width: int) -> str:
    """
    If idx is a pure integer string, zero-pad it to `width`.
    Otherwise return idx as-is.
    """
    s = str(idx).strip()
    return s.zfill(width) if s.isdigit() else s

def safe_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def download_music(index_raw: str, url: str, base_dir: str, index_width: int) -> None:
    index = zero_pad_index(index_raw, index_width)

    # songs/0001/
    output_dir = os.path.join(base_dir, index)
    os.makedirs(output_dir, exist_ok=True)

    outtmpl = os.path.join(output_dir, f"{index}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "js_runtimes": {
            "node": {}
        },

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],

        "restrictfilenames": True,
        "trim_file_name": 180,

        "quiet": False,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    meta = {
        "index_raw": index_raw,
        "index": index,
        "source_url": url,

        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "channel": info.get("channel"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "webpage_url": info.get("webpage_url"),

        "extractor": info.get("extractor"),
        "ext_downloaded": info.get("ext"),
        "audio_channels": info.get("audio_channels"),
        "abr": info.get("abr"),
        "asr": info.get("asr"),
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    safe_write_json(meta_path, meta)

def main():
    with open("data/MIR-CE500_link.json", "r", encoding="utf-8") as f:
        music_link = json.load(f)

    numeric_keys = [str(k).strip() for k in music_link.keys() if str(k).strip().isdigit()]
    if numeric_keys:
        max_len = max(len(k) for k in numeric_keys)
        index_width = max(4, max_len)
    else:
        index_width = 4

    base_dir = "songs"
    os.makedirs(base_dir, exist_ok=True)

    failed = []

    for idx, url in music_link.items():
        idx_str = str(idx)

        try:
            download_music(idx_str, str(url), base_dir=base_dir, index_width=index_width)
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            padded_idx = idx_str.zfill(index_width) if idx_str.isdigit() else idx_str

            print(f"[FAIL] index={padded_idx} url={url}")
            print(f"       error={e}")

            failed.append({
                "index_raw": idx_str,
                "index": padded_idx,
                "url": url,
                "error_type": type(e).__name__,
                "error_message": str(e),
            })

            continue

    if failed:
        failed_path = os.path.join(base_dir, "failed_downloads.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

        print(f"\n Failed downloads saved to: {failed_path}")
    else:
        print("\n All downloads completed successfully.")

if __name__ == "__main__":
    main()