"""
Train frame-level chord recognition (CRNN on precomputed chroma).
Run from project root: python train.py
  or: python train.py --config config.yaml
  Colab: set data_root in YAML to your Drive path (see config.colab.example.yaml).
Writes training_history.json each epoch for plotting in a notebook.
"""
import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import IGNORE_INDEX, ChordDataset, get_num_classes
from models.crnn import ChordCRNN


def _resolve_under_root(root: Path, p: str) -> str:
    path = Path(p)
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def load_yaml_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("Install PyYAML for --config: pip install pyyaml") from e
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def build_train_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Merge optional YAML config with CLI (CLI wins when flag is explicitly set)."""
    cfg: dict = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)

    root = Path(cfg.get("data_root", ".")).resolve()
    if args.data_root is not None:
        root = Path(args.data_root).resolve()

    paths = cfg.get("paths", {})
    ns = argparse.Namespace()
    ns.data_root = root
    ns.splits_path = _resolve_under_root(root, paths.get("splits", "data/splits.json"))
    ns.vocab_path = _resolve_under_root(root, paths.get("vocabulary", "data/chord_vocabulary.json"))
    ns.songs_dir = _resolve_under_root(root, paths.get("songs", "songs"))

    tc = cfg.get("train", {})
    ns.reduced = str(args.reduced if args._cli_reduced else tc.get("reduced", "25"))
    ns.segment_frames = args.segment_frames if args._cli_segment_frames else int(tc.get("segment_frames", 200))
    ns.batch_size = args.batch_size if args._cli_batch_size else int(tc.get("batch_size", 16))
    ns.epochs = args.epochs if args._cli_epochs else int(tc.get("epochs", 30))
    ns.lr = args.lr if args._cli_lr else float(tc.get("lr", 1e-3))
    ns.weight_decay = args.weight_decay if args._cli_weight_decay else float(tc.get("weight_decay", 1e-4))
    ns.num_workers = args.num_workers if args._cli_num_workers else int(tc.get("num_workers", 0))
    dev = tc.get("device", None)
    ns.device = args.device if args._cli_device else (dev if dev is not None else None)

    oc = cfg.get("output", {})
    save_default = Path(oc.get("checkpoint", "checkpoints/chord_crnn.pt"))
    hist_default = Path(oc.get("history", "checkpoints/training_history.json"))
    ns.save = args.save if args._cli_save else (root / save_default if not save_default.is_absolute() else save_default)
    ns.history = args.history if args._cli_history else (root / hist_default if not hist_default.is_absolute() else hist_default)
    if not isinstance(ns.save, Path):
        ns.save = Path(ns.save)
    if not isinstance(ns.history, Path):
        ns.history = Path(ns.history)
    ns.save = ns.save.resolve()
    ns.history = ns.history.resolve()

    return ns


def parse_args():
    p = argparse.ArgumentParser(description="Train chord CRNN (optional config.yaml for paths)")
    p.add_argument("--config", type=Path, default=None, help="YAML config (data_root, paths, train, output)")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override config data_root: folder containing data/ and songs/ (e.g. Colab Drive path)",
    )
    p.add_argument("--reduced", choices=("full", "25", "61"), default=None, help="label space")
    p.add_argument("--segment-frames", type=int, default=None, help="fixed time length per batch item")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--save", type=Path, default=None)
    p.add_argument(
        "--history",
        type=Path,
        default=None,
        help="JSON with per-epoch train/val loss & acc",
    )
    args = p.parse_args()
    args._cli_reduced = args.reduced is not None
    args._cli_segment_frames = args.segment_frames is not None
    args._cli_batch_size = args.batch_size is not None
    args._cli_epochs = args.epochs is not None
    args._cli_lr = args.lr is not None
    args._cli_weight_decay = args.weight_decay is not None
    args._cli_num_workers = args.num_workers is not None
    args._cli_device = args.device is not None
    args._cli_save = args.save is not None
    args._cli_history = args.history is not None
    if args.config is None:
        default_cfg = ROOT / "config.yaml"
        if default_cfg.is_file():
            args.config = default_cfg
    if args.reduced is None:
        args.reduced = "25"
    if args.segment_frames is None:
        args.segment_frames = 200
    if args.batch_size is None:
        args.batch_size = 16
    if args.epochs is None:
        args.epochs = 30
    if args.lr is None:
        args.lr = 1e-3
    if args.weight_decay is None:
        args.weight_decay = 1e-4
    if args.num_workers is None:
        args.num_workers = 0
    if args.save is None:
        args.save = Path("checkpoints/chord_crnn.pt")
    if args.history is None:
        args.history = Path("checkpoints/training_history.json")
    return args


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    mask = labels != IGNORE_INDEX
    if not mask.any():
        return 0.0
    return (pred[mask] == labels[mask]).float().mean().item()


def write_history(path: Path, meta: dict, epochs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "epochs": epochs}, f, ensure_ascii=False, indent=2)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        n_batches += 1
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main():
    args_cli = parse_args()
    args = build_train_namespace(args_cli)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"data_root: {args.data_root}")
    print(f"splits: {args.splits_path}")
    print(f"vocabulary: {args.vocab_path}")
    print(f"songs: {args.songs_dir}")

    reduced_25 = args.reduced == "25"
    reduced_61 = args.reduced == "61"

    train_ds = ChordDataset(
        "train",
        splits_path=args.splits_path,
        songs_dir=args.songs_dir,
        segment_frames=args.segment_frames,
        reduced_25=reduced_25,
        reduced_61=reduced_61,
        vocab_path=args.vocab_path,
    )
    val_ds = ChordDataset(
        "val",
        splits_path=args.splits_path,
        songs_dir=args.songs_dir,
        segment_frames=args.segment_frames,
        reduced_25=reduced_25,
        reduced_61=reduced_61,
        vocab_path=args.vocab_path,
    )
    if len(train_ds) == 0:
        print("No training samples (missing features.npy/labels.npy or empty split). Run preprocess_audio.py first.")
        return

    num_classes = get_num_classes(reduced_25=reduced_25, reduced_61=reduced_61, vocab_path=args.vocab_path)
    model = ChordCRNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_root": str(args.data_root),
        "reduced": args.reduced,
        "segment_frames": args.segment_frames,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_classes": num_classes,
        "device": str(device),
    }
    history_epochs: list = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"epoch {epoch:3d}  train loss {tr_loss:.4f} acc {tr_acc:.4f}  val loss {va_loss:.4f} acc {va_acc:.4f}")
        history_epochs.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "train_acc": round(tr_acc, 6),
                "val_loss": round(va_loss, 6),
                "val_acc": round(va_acc, 6),
            }
        )
        write_history(args.history, meta, history_epochs)
        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": num_classes,
                    "reduced": args.reduced,
                    "segment_frames": args.segment_frames,
                },
                args.save,
            )
            print(f"  saved {args.save}")

    print(f"History written to {args.history}")


if __name__ == "__main__":
    main()
