"""
Train frame-level chord recognition (CRNN on precomputed frame features).
Run from project root: python train.py
  or: python train.py --config config.yaml
  Colab: set data_root + output.dir (e.g. Drive checkpoints folder); ckpt/history get a datetime suffix.
  Resume: python train.py --config ... --resume path/to/chord_crnn_cqt84_YYYYMMDD.pt
    (loads model weights only; optimizer starts fresh. New run_stamp / history / save_path.)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import IGNORE_INDEX, ChordDataset, get_num_classes
from models.crnn import ChordCRNN, default_chord_crnn_kwargs


def _resolve_under_root(root: Path, p: str) -> str:
    path = Path(p)
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def load_class_weight_tensor(
    path: Path,
    num_classes: int,
    scheme: str,
    eps: float,
    clip_max: Optional[float],
) -> torch.Tensor:
    """
    Load per-class CE weights from JSON: either a 'weights' list or 'counts' with scheme inv_sqrt / inv_freq.
    Weights are normalized so arithmetic mean equals 1; optional clip_max then re-normalize.
    """
    scheme = scheme.lower().strip()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "weights" in data:
        w = torch.tensor(data["weights"], dtype=torch.float64)
    elif "counts" in data:
        if scheme in ("", "none"):
            raise ValueError(
                f"{path}: JSON has 'counts' but class_weight_scheme is none; set inv_sqrt or inv_freq."
            )
        counts = torch.tensor(data["counts"], dtype=torch.float64)
        if scheme == "inv_sqrt":
            w = 1.0 / torch.sqrt(torch.clamp(counts, min=1.0))
        elif scheme == "inv_freq":
            w = 1.0 / (counts + float(eps))
        else:
            raise ValueError(f"Unknown class_weight_scheme: {scheme}")
    else:
        raise ValueError(f"{path}: expected 'counts' or 'weights' in JSON")

    if w.numel() != num_classes:
        raise ValueError(f"{path}: expected {num_classes} weights, got {w.numel()}")
    w = w * (num_classes / w.sum())
    if clip_max is not None and clip_max > 0:
        w = torch.clamp(w, max=float(clip_max))
        w = w * (num_classes / w.sum())
    return w.to(dtype=torch.float32)


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
    ns.input_dim = int(args.input_dim if args._cli_input_dim else int(tc.get("input_dim", 12)))
    ns.segment_frames = args.segment_frames if args._cli_segment_frames else int(tc.get("segment_frames", 200))
    if args._cli_segment_hop_frames:
        ns.segment_hop_frames = int(args.segment_hop_frames)
    else:
        sh = tc.get("segment_hop_frames")
        ns.segment_hop_frames = int(sh) if sh is not None else ns.segment_frames
    ns.batch_size = args.batch_size if args._cli_batch_size else int(tc.get("batch_size", 16))
    ns.epochs = args.epochs if args._cli_epochs else int(tc.get("epochs", 30))
    ns.lr = args.lr if args._cli_lr else float(tc.get("lr", 1e-3))
    ns.weight_decay = args.weight_decay if args._cli_weight_decay else float(tc.get("weight_decay", 3e-4))
    ns.dropout = float(args.dropout if args._cli_dropout else tc.get("dropout", 0.4))
    ns.scheduler = str(args.scheduler if args._cli_scheduler else tc.get("scheduler", "plateau")).lower()
    ns.scheduler_patience = int(
        args.scheduler_patience if args._cli_scheduler_patience else tc.get("scheduler_patience", 4)
    )
    ns.scheduler_factor = float(
        args.scheduler_factor if args._cli_scheduler_factor else tc.get("scheduler_factor", 0.5)
    )
    ns.scheduler_min_lr = float(
        args.scheduler_min_lr if args._cli_scheduler_min_lr else float(tc.get("scheduler_min_lr", 1e-6))
    )
    ns.num_workers = args.num_workers if args._cli_num_workers else int(tc.get("num_workers", 0))
    dev = tc.get("device", None)
    ns.device = args.device if args._cli_device else (dev if dev is not None else None)

    if args._cli_no_augment_pitch:
        ns.augment_pitch = False
    else:
        ns.augment_pitch = bool(tc.get("augment_pitch", True))
    ns.augment_prob = (
        float(args.augment_prob)
        if args._cli_augment_prob
        else float(tc.get("augment_prob", 0.5))
    )
    ns.pitch_shift_max = (
        int(args.pitch_shift_max)
        if args._cli_pitch_shift_max
        else int(tc.get("pitch_shift_max", 5))
    )

    oc = cfg.get("output", {})
    if "dir" in oc:
        out_dir_raw = oc["dir"]
    elif oc.get("checkpoint"):
        out_dir_raw = str(Path(oc["checkpoint"]).parent)
    else:
        out_dir_raw = "checkpoints"
    out_dir_p = Path(out_dir_raw)
    ns.output_dir = out_dir_p.resolve() if out_dir_p.is_absolute() else (root / out_dir_p).resolve()
    ns.checkpoint_prefix = oc.get("checkpoint_prefix", "chord_crnn")
    ns.history_prefix = oc.get("history_prefix", "training_history")
    if args._cli_output_dir:
        ns.output_dir = Path(args.output_dir).resolve()

    if args.resume is not None:
        ns.resume_path = Path(args.resume).expanduser().resolve()
    else:
        rp = tc.get("resume")
        if rp:
            p = Path(str(rp).strip())
            ns.resume_path = p.resolve() if p.is_absolute() else (root / p).resolve()
        else:
            ns.resume_path = None

    cw_cli = args._cli_class_weight_json
    cw_raw = args.class_weight_json if cw_cli else tc.get("class_weight_json")
    if cw_raw is None or (isinstance(cw_raw, str) and cw_raw.strip().lower() in ("", "none", "null")):
        ns.class_weight_path = None
    else:
        p = Path(str(cw_raw).strip())
        ns.class_weight_path = p.resolve() if p.is_absolute() else (root / p).resolve()

    ns.class_weight_scheme = (
        str(args.class_weight_scheme).lower().strip()
        if args.class_weight_scheme is not None
        else str(tc.get("class_weight_scheme", "none")).lower().strip()
    )
    ns.class_weight_eps = float(
        args.class_weight_eps if args._cli_class_weight_eps else float(tc.get("class_weight_eps", 1.0))
    )
    if args._cli_class_weight_clip_max:
        ns.class_weight_clip_max = float(args.class_weight_clip_max)
    else:
        cm = tc.get("class_weight_clip_max")
        ns.class_weight_clip_max = float(cm) if cm is not None else None

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
    p.add_argument(
        "--segment-hop-frames",
        type=int,
        default=None,
        help="stride between neighboring chunks (default: segment_frames; set 100 for 50% overlap when segment_frames=200)",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None, help="ChordCRNN dropout (default 0.4)")
    p.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Feature size per frame (12 for chroma features.npy, 84 for cqt84)",
    )
    p.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default=None,
        help="LR schedule: ReduceLROnPlateau on val loss, or none",
    )
    p.add_argument("--scheduler-patience", type=int, default=None, help="plateau epochs before lr reduce")
    p.add_argument("--scheduler-factor", type=float, default=None, help="lr *= factor on plateau")
    p.add_argument("--scheduler-min-lr", type=float, default=None, help="minimum lr for plateau")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--no-augment-pitch",
        action="store_true",
        help="disable train-time pitch roll + reduced_25 label rotation (overrides config)",
    )
    p.add_argument("--augment-prob", type=float, default=None, help="probability of pitch shift per sample (train)")
    p.add_argument(
        "--pitch-shift-max",
        type=int,
        default=None,
        help="max |semitones| for pitch shift (integer roll; excludes 0)",
    )
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save checkpoint + history (default: config output.dir or data_root/checkpoints)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Load model weights from a prior .pt (expects 'model' state_dict); optimizer not restored",
    )
    p.add_argument(
        "--class-weight-json",
        type=Path,
        default=None,
        help="JSON with 'counts' (see scripts/count_reduced25_train_frames.py) or explicit 'weights' list",
    )
    p.add_argument(
        "--class-weight-scheme",
        choices=("none", "inv_sqrt", "inv_freq"),
        default=None,
        help="With counts: inv_sqrt or inv_freq; none disables (unless JSON has 'weights')",
    )
    p.add_argument(
        "--class-weight-eps",
        type=float,
        default=None,
        help="Smoothing for inv_freq: w ∝ 1/(count+eps) (default from config or 1.0)",
    )
    p.add_argument(
        "--class-weight-clip-max",
        type=float,
        default=None,
        help="After normalizing mean=1, clamp each weight to at most this (optional)",
    )
    p.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    args = p.parse_args()
    args._cli_reduced = args.reduced is not None
    args._cli_segment_frames = args.segment_frames is not None
    args._cli_segment_hop_frames = args.segment_hop_frames is not None
    args._cli_batch_size = args.batch_size is not None
    args._cli_epochs = args.epochs is not None
    args._cli_lr = args.lr is not None
    args._cli_weight_decay = args.weight_decay is not None
    args._cli_dropout = args.dropout is not None
    args._cli_input_dim = args.input_dim is not None
    args._cli_scheduler = args.scheduler is not None
    args._cli_scheduler_patience = args.scheduler_patience is not None
    args._cli_scheduler_factor = args.scheduler_factor is not None
    args._cli_scheduler_min_lr = args.scheduler_min_lr is not None
    args._cli_num_workers = args.num_workers is not None
    args._cli_device = args.device is not None
    args._cli_output_dir = args.output_dir is not None
    args._cli_no_augment_pitch = bool(args.no_augment_pitch)
    args._cli_augment_prob = args.augment_prob is not None
    args._cli_pitch_shift_max = args.pitch_shift_max is not None
    args._cli_class_weight_json = args.class_weight_json is not None
    args._cli_class_weight_scheme = args.class_weight_scheme is not None
    args._cli_class_weight_eps = args.class_weight_eps is not None
    args._cli_class_weight_clip_max = args.class_weight_clip_max is not None
    if args.config is None:
        default_cfg = ROOT / "config.yaml"
        if default_cfg.is_file():
            args.config = default_cfg
    if args.reduced is None:
        args.reduced = "25"
    if args.segment_frames is None:
        args.segment_frames = 200
    if args.segment_hop_frames is None:
        args.segment_hop_frames = args.segment_frames
    if args.batch_size is None:
        args.batch_size = 16
    if args.epochs is None:
        args.epochs = 30
    if args.lr is None:
        args.lr = 1e-3
    if args.weight_decay is None:
        args.weight_decay = 3e-4
    if args.dropout is None:
        args.dropout = 0.4
    if args.scheduler is None:
        args.scheduler = "plateau"
    if args.scheduler_patience is None:
        args.scheduler_patience = 4
    if args.scheduler_factor is None:
        args.scheduler_factor = 0.5
    if args.scheduler_min_lr is None:
        args.scheduler_min_lr = 1e-6
    if args.num_workers is None:
        args.num_workers = 0
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


def load_model_weights_from_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    model_kw: dict,
) -> dict:
    """
    Load ChordCRNN weights from torch.save dict. Validates num_classes and input_dim vs checkpoint meta.
    Returns the raw checkpoint dict (for logging).
    """
    if not path.is_file():
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Invalid checkpoint (expected dict with 'model' key): {path}")
    saved_nc = ckpt.get("num_classes")
    if saved_nc is not None and int(saved_nc) != int(num_classes):
        raise ValueError(
            f"Checkpoint num_classes={saved_nc} does not match current run num_classes={num_classes}"
        )
    mkw = ckpt.get("model_init_kwargs") or {}
    if mkw:
        if int(mkw.get("num_classes", num_classes)) != int(num_classes):
            raise ValueError("Checkpoint model_init_kwargs.num_classes mismatch")
        if int(mkw.get("input_dim", model_kw["input_dim"])) != int(model_kw["input_dim"]):
            raise ValueError(
                f"Checkpoint input_dim={mkw.get('input_dim')} does not match current input_dim={model_kw['input_dim']}"
            )
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


def _namespace_to_json_dict(ns: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(ns).items():
        if k.startswith("_"):
            continue
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _cli_args_for_json(cli: argparse.Namespace) -> dict:
    skip_prefix = "_"
    out = {}
    for k, v in vars(cli).items():
        if k.startswith(skip_prefix):
            continue
        if isinstance(v, Path):
            out[k] = str(v) if v is not None else None
        else:
            out[k] = v
    return out


def _model_arch_dict(model: nn.Module, init_kwargs: dict) -> dict:
    kw = {}
    for k, v in init_kwargs.items():
        kw[k] = list(v) if isinstance(v, tuple) else v
    return {
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "init_kwargs": kw,
        "repr": str(model),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    desc: str = "",
    disable_progress: bool = False,
):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    it = loader if disable_progress else tqdm(loader, desc=desc, leave=True)
    for x, y in it:
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
        batch_acc = accuracy(logits, y)
        total_acc += batch_acc
        n_batches += 1
        if not disable_progress and hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main():
    args_cli = parse_args()
    args = build_train_namespace(args_cli)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started_iso = datetime.now().isoformat(timespec="seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f"{args.checkpoint_prefix}_{run_stamp}.pt"
    history_path = args.output_dir / f"{args.history_prefix}_{run_stamp}.json"

    print(f"data_root: {args.data_root}")
    print(f"splits: {args.splits_path}")
    print(f"vocabulary: {args.vocab_path}")
    print(f"songs: {args.songs_dir}")
    print(f"input_dim: {args.input_dim}")
    print(f"segment_frames: {args.segment_frames}, segment_hop_frames: {args.segment_hop_frames}")

    reduced_25 = args.reduced == "25"
    reduced_61 = args.reduced == "61"
    if reduced_25:
        print(
            f"augment_pitch: {args.augment_pitch} "
            f"(prob={args.augment_prob}, pitch_shift_max={args.pitch_shift_max}, train only)"
        )
    print(f"checkpoint: {save_path}")
    print(f"history: {history_path}")
    if args.resume_path is not None:
        print(f"resume: {args.resume_path}")

    train_ds = ChordDataset(
        "train",
        splits_path=args.splits_path,
        songs_dir=args.songs_dir,
        segment_frames=args.segment_frames,
        segment_hop_frames=args.segment_hop_frames,
        reduced_25=reduced_25,
        reduced_61=reduced_61,
        vocab_path=args.vocab_path,
        augment_pitch=args.augment_pitch and reduced_25,
        augment_prob=args.augment_prob,
        pitch_shift_max=args.pitch_shift_max,
    )
    val_ds = ChordDataset(
        "val",
        splits_path=args.splits_path,
        songs_dir=args.songs_dir,
        segment_frames=args.segment_frames,
        segment_hop_frames=args.segment_hop_frames,
        reduced_25=reduced_25,
        reduced_61=reduced_61,
        vocab_path=args.vocab_path,
    )
    if len(train_ds) == 0:
        print("No training samples (missing features.npy/labels.npy or empty split). Run preprocess_audio.py first.")
        return

    num_classes = get_num_classes(reduced_25=reduced_25, reduced_61=reduced_61, vocab_path=args.vocab_path)
    model_kw = default_chord_crnn_kwargs(num_classes, dropout=args.dropout, input_dim=args.input_dim)
    model = ChordCRNN(**model_kw).to(device)
    resume_ckpt = None
    if args.resume_path is not None:
        resume_ckpt = load_model_weights_from_checkpoint(
            args.resume_path,
            model,
            device,
            num_classes=num_classes,
            model_kw=model_kw,
        )
        print(f"Loaded model weights from {args.resume_path}")

    ce_weight: Optional[torch.Tensor] = None
    if args.class_weight_path is not None:
        ce_weight = load_class_weight_tensor(
            args.class_weight_path,
            num_classes,
            args.class_weight_scheme,
            args.class_weight_eps,
            args.class_weight_clip_max,
        ).to(device)
        print(
            f"class_weight: {args.class_weight_path} scheme={args.class_weight_scheme!r} "
            f"mean={ce_weight.mean().item():.4f} min={ce_weight.min().item():.4f} max={ce_weight.max().item():.4f}"
        )
    criterion = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )
        print(
            f"scheduler: ReduceLROnPlateau(patience={args.scheduler_patience}, "
            f"factor={args.scheduler_factor}, min_lr={args.scheduler_min_lr})"
        )
    else:
        print("scheduler: none")

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

    meta = {
        "data_root": str(args.data_root),
        "reduced": args.reduced,
        "segment_frames": args.segment_frames,
        "segment_hop_frames": args.segment_hop_frames,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "input_dim": args.input_dim,
        "scheduler": args.scheduler,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_min_lr": args.scheduler_min_lr,
        "augment_pitch": args.augment_pitch and reduced_25,
        "augment_prob": args.augment_prob,
        "pitch_shift_max": args.pitch_shift_max,
        "class_weight_path": str(args.class_weight_path) if args.class_weight_path else None,
        "class_weight_scheme": args.class_weight_scheme,
        "class_weight_eps": args.class_weight_eps,
        "class_weight_clip_max": args.class_weight_clip_max,
        "class_weights": ce_weight.detach().cpu().tolist() if ce_weight is not None else None,
        "num_classes": num_classes,
        "device": str(device),
        "run_stamp": run_stamp,
        "run_started_iso": run_started_iso,
        "save_path": str(save_path),
        "history_path": str(history_path),
        "resume_from": str(args.resume_path) if args.resume_path else None,
        "resume_run_stamp": (resume_ckpt.get("run_stamp") if resume_ckpt else None),
        "config_path": str(args_cli.config) if args_cli.config else None,
        "args_cli": _cli_args_for_json(args_cli),
        "args_merged": _namespace_to_json_dict(args),
        "model": _model_arch_dict(model, model_kw),
    }
    history_epochs: list = []
    best_val = float("inf")
    show = not args_cli.no_progress
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            desc=f"train e{epoch}/{args.epochs}",
            disable_progress=not show,
        )
        va_loss, va_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            desc=f"val   e{epoch}/{args.epochs}",
            disable_progress=not show,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(va_loss)
        print(
            f"epoch {epoch:3d}  lr {current_lr:.2e}  "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  val loss {va_loss:.4f} acc {va_acc:.4f}"
        )
        history_epochs.append(
            {
                "epoch": epoch,
                "lr": round(current_lr, 10),
                "train_loss": round(tr_loss, 6),
                "train_acc": round(tr_acc, 6),
                "val_loss": round(va_loss, 6),
                "val_acc": round(va_acc, 6),
            }
        )
        write_history(history_path, meta, history_epochs)
        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": num_classes,
                    "reduced": args.reduced,
                    "segment_frames": args.segment_frames,
                    "segment_hop_frames": args.segment_hop_frames,
                    "run_stamp": run_stamp,
                    "meta": meta,
                    "model_class": "ChordCRNN",
                    "model_init_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_kw.items()},
                    "model_repr": str(model),
                },
                save_path,
            )
            print(f"  saved {save_path}")

    print(f"History written to {history_path}")


if __name__ == "__main__":
    main()
