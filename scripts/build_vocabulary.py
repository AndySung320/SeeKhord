"""
Build chord vocabulary from annotations (full corrected).
Output: data/chord_vocabulary.json with vocabulary, id_to_symbol, num_classes,
and reductions as "original chord -> reduced chord symbol" (25-class and 61-class),
plus reduction_25 / reduction_61 id lists for Dataset.
"""
import json
import os
import re
from typing import Optional

CORRECTED_PATH = "data/MIR-CE500_corrected.json"
OUTPUT_PATH = "data/chord_vocabulary.json"
OUTPUT_REDUCED_61 = "data/chord_reduced_61.json"

ROOTS_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOT_FLAT_TO_SHARP = {
    "Cb": "B", "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#",
}
QUALITIES_61 = ("maj", "maj7", "7", "min", "min7")
NUM_REDUCED_25 = 25   # N + 12 maj + 12 min
NUM_REDUCED_61 = 61   # N + 12 * 5 (maj, maj7, 7, min, min7)


def _normalize_root(root_str: str) -> Optional[str]:
    """Return root in sharp form (e.g. Bb -> A#), or None."""
    root_str = root_str.strip()
    if root_str in ROOTS_12:
        return root_str
    if root_str in ROOT_FLAT_TO_SHARP:
        return ROOT_FLAT_TO_SHARP[root_str]
    return None


def _root_to_index(root_str: str) -> Optional[int]:
    """Normalize root to 0..11."""
    r = _normalize_root(root_str)
    return ROOTS_12.index(r) if r is not None else None


def _parse_chord(chord_symbol: str) -> tuple[Optional[str], str]:
    """Return (normalized_root_str, quality_str). quality_str lower, root None if N or unparseable."""
    chord_symbol = chord_symbol.strip()
    if chord_symbol == "N":
        return None, "N"
    m = re.match(r"^([A-G][#b]?):(.+)$", chord_symbol, re.IGNORECASE)
    if m:
        root_str, quality = m.group(1), m.group(2).strip().lower()
        root = _normalize_root(root_str)
        return root, quality
    root = _normalize_root(chord_symbol)
    if root is not None:
        return root, "maj"  # root only -> major
    return None, "N"


def symbol_to_reduced_25(chord_symbol: str) -> str:
    """Map original chord -> reduced chord symbol (25-class). Returns 'N' or 'Root:maj' / 'Root:min'."""
    root, quality = _parse_chord(chord_symbol)
    if root is None:
        return "N"
    # Minor only when clearly minor (avoid 'min' substring in 'dominant', and quality[0]=='m' matching 'maj')
    if quality == "m" or quality.startswith("min") or ("dim" in quality and "7" not in quality):
        return f"{root}:min"
    return f"{root}:maj"


def _quality_to_61(quality: str) -> str:
    """Map raw quality string to one of maj, maj7, 7, min, min7 (for 61-class)."""
    q = quality.lower().strip()
    if not q or q == "n":
        return "maj"
    # min7 family first (before min): min7, m7, hdim7, dim7, half-dim
    if "min7" in q or "m7" in q or "hdim7" in q or "halfdim" in q or "hdim" in q or ("dim" in q and "7" in q):
        return "min7"
    # min triad: min, m, minor, dim (no 7)
    if "min" in q or q == "m" or ("dim" in q and "7" not in q):
        return "min"
    # maj7: maj7, ma7, major7, maj9, Δ
    if "maj7" in q or "ma7" in q or "major7" in q or "maj9" in q or "δ" in q or "delta" in q:
        return "maj7"
    # dominant 7: 7, dom7 (standalone 7 or dom)
    if q == "7" or "dom7" in q or "dominant7" in q or (q.startswith("7") and "maj" not in q and "min" not in q):
        return "7"
    # maj: maj, major, 5, 6, aug, sus2, sus4, add9, (1,5), slash, etc.
    return "maj"


def symbol_to_reduced_61(chord_symbol: str) -> str:
    """Map original chord -> reduced chord symbol (61-class). Returns 'N' or 'Root:quality' in {maj, maj7, 7, min, min7}."""
    root, quality = _parse_chord(chord_symbol)
    if root is None:
        return "N"
    q61 = _quality_to_61(quality)
    return f"{root}:{q61}"


def main():
    with open(CORRECTED_PATH, "r", encoding="utf-8") as f:
        corrected = json.load(f)

    chord_set = set()
    for idx in corrected:
        for seg in corrected[idx]:
            chord_set.add(seg[2].strip())

    rest = sorted(chord_set - {"N"})
    vocab = {"N": 0}
    for i, c in enumerate(rest, start=1):
        vocab[c] = i
    num_classes = len(vocab)
    id_to_symbol = {v: k for k, v in vocab.items()}

    # Reduced symbol (original -> reduced chord symbol)
    reduced_symbol_25 = {sym: symbol_to_reduced_25(sym) for sym in vocab}
    reduced_symbol_61 = {sym: symbol_to_reduced_61(sym) for sym in vocab}

    # Ordered reduced symbols for stable id
    order_25 = ["N"] + [f"{r}:maj" for r in ROOTS_12] + [f"{r}:min" for r in ROOTS_12]
    reduced_symbol_to_id_25 = {s: i for i, s in enumerate(order_25)}
    order_61 = ["N"] + [f"{r}:{q}" for r in ROOTS_12 for q in QUALITIES_61]
    reduced_symbol_to_id_61 = {s: i for i, s in enumerate(order_61)}

    reduction_25 = [reduced_symbol_to_id_25.get(reduced_symbol_25[id_to_symbol[i]], 0) for i in range(num_classes)]
    reduction_61 = [reduced_symbol_to_id_61.get(reduced_symbol_61[id_to_symbol[i]], 0) for i in range(num_classes)]

    out = {
        "vocabulary": vocab,
        "id_to_symbol": id_to_symbol,
        "num_classes": num_classes,
        "reduced_symbol_25": reduced_symbol_25,
        "reduced_symbol_61": reduced_symbol_61,
        "reduced_symbol_to_id_25": reduced_symbol_to_id_25,
        "reduced_symbol_to_id_61": reduced_symbol_to_id_61,
        "reduction_25": reduction_25,
        "reduction_61": reduction_61,
        "num_classes_reduced_25": NUM_REDUCED_25,
        "num_classes_reduced_61": NUM_REDUCED_61,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out_61 = {
        "reduced_symbol_61": reduced_symbol_61,
        "reduced_symbol_to_id_61": reduced_symbol_to_id_61,
        "num_classes": NUM_REDUCED_61,
    }
    with open(OUTPUT_REDUCED_61, "w", encoding="utf-8") as f:
        json.dump(out_61, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUTPUT_PATH}: {num_classes} full classes, 25-class (maj/min), 61-class (maj/maj7/7/min/min7).")
    print(f"Wrote {OUTPUT_REDUCED_61}: reduced_symbol_61 dictionary and id map.")


if __name__ == "__main__":
    main()
