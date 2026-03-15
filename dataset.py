"""
PyTorch Dataset for chord recognition: loads precomputed features and frame-level labels
from songs/<index>/features.npy and labels.npy according to data/splits.json.
Labels on disk are always full-vocab IDs; use reduced_25=True or reduced_61=True to get
reduced classes without re-running preprocess.
"""
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS_PATH = "data/splits.json"
VOCAB_PATH = "data/chord_vocabulary.json"
SONGS_DIR = "songs"
IGNORE_INDEX = -100


class ChordDataset(Dataset):
    """
    - split: "train" | "val" | "test"
    - segment_frames: if set, each item is a fixed-length chunk (with padding); else full song.
    - reduced_25: 25-class (N + 12 maj + 12 min).
    - reduced_61: 61-class (N + 12 roots x {maj, maj7, 7, min, min7}). Ignored if reduced_25 is True.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        splits_path: str = SPLITS_PATH,
        songs_dir: str = SONGS_DIR,
        segment_frames: Optional[int] = None,
        reduced_25: bool = False,
        reduced_61: bool = False,
        vocab_path: str = VOCAB_PATH,
    ):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.indices = splits[split]
        self.songs_dir = Path(songs_dir)
        self.segment_frames = segment_frames
        self.reduced_25 = reduced_25
        self.reduced_61 = reduced_61 and not reduced_25
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.reduction_25 = vocab_data.get("reduction_25")
        self.reduction_61 = vocab_data.get("reduction_61")
        if self.reduced_25 and not self.reduction_25:
            raise ValueError("reduced_25=True but chord_vocabulary.json has no reduction_25; run scripts/build_vocabulary.py")
        if self.reduced_61 and not self.reduction_61:
            raise ValueError("reduced_61=True but chord_vocabulary.json has no reduction_61; run scripts/build_vocabulary.py")
        self._segments: list[tuple[str, int, int]] = []
        for index in self.indices:
            feat_path = self.songs_dir / index / "features.npy"
            label_path = self.songs_dir / index / "labels.npy"
            if not feat_path.is_file() or not label_path.is_file():
                continue
            feat = np.load(feat_path)
            T = feat.shape[0]
            if segment_frames is None:
                self._segments.append((index, 0, T))
            else:
                for start in range(0, T, segment_frames):
                    end = min(start + segment_frames, T)
                    self._segments.append((index, start, end))

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        index, start, end = self._segments[idx]
        feat_path = self.songs_dir / index / "features.npy"
        label_path = self.songs_dir / index / "labels.npy"
        features = np.load(feat_path)
        labels = np.load(label_path)
        features = features[start:end]
        labels = labels[start:end].copy()
        if self.reduced_25 and self.reduction_25:
            reduction = self.reduction_25
        elif self.reduced_61 and self.reduction_61:
            reduction = self.reduction_61
        else:
            reduction = None
        if reduction is not None:
            n_classes = len(reduction)
            valid = (labels >= 0) & (labels < n_classes)
            reduced = np.full_like(labels, IGNORE_INDEX)
            reduced[valid] = np.array(reduction, dtype=np.int64)[labels[valid]]
            labels = reduced
        if self.segment_frames is not None and features.shape[0] < self.segment_frames:
            pad = self.segment_frames - features.shape[0]
            features = np.pad(
                features,
                ((0, pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            labels = np.pad(
                labels,
                (0, pad),
                mode="constant",
                constant_values=IGNORE_INDEX,
            )
        return torch.from_numpy(features).float(), torch.from_numpy(labels).long()


def get_num_classes(reduced_25: bool = False, reduced_61: bool = False, vocab_path: str = VOCAB_PATH) -> int:
    with open(vocab_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if reduced_25 and d.get("num_classes_reduced_25") is not None:
        return int(d["num_classes_reduced_25"])
    if reduced_61 and d.get("num_classes_reduced_61") is not None:
        return int(d["num_classes_reduced_61"])
    return int(d["num_classes"])
