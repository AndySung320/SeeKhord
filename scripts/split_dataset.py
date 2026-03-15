"""
Split trainable song indices into train / val / test (80 / 10 / 10).
Output: data/splits.json with fixed random seed for reproducibility.
"""
import json
import os
import random

TRAINABLE_PATH = "data/trainable_songs.json"
OUTPUT_PATH = "data/splits.json"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test = 1 - train - val


def main():
    with open(TRAINABLE_PATH, "r", encoding="utf-8") as f:
        trainable = json.load(f)
    indices = [e["index"] for e in trainable]
    if not indices:
        out = {"seed": SEED, "train": [], "val": [], "test": []}
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote {OUTPUT_PATH}: no trainable songs.")
        return
    random.seed(SEED)
    random.shuffle(indices)
    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    train_list = indices[:n_train]
    val_list = indices[n_train : n_train + n_val]
    test_list = indices[n_train + n_val :]
    out = {
        "seed": SEED,
        "train": train_list,
        "val": val_list,
        "test": test_list,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUTPUT_PATH}: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}.")


if __name__ == "__main__":
    main()
