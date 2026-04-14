"""
Frame-level chord classifiers:
- ChordCRNN: 1D CNN + BiLSTM + per-frame logits.
- ChordCRNNAttention: 1D CNN + BiLSTM + self-attention + per-frame logits.
- ChordCRNNMultiscale: multi-kernel parallel Conv1d + 1x1 fusion + BiLSTM + per-frame logits.
Input: (batch, time, F) features from preprocess (F=12 chroma or F=84 log-CQT, etc.).
"""
from typing import Tuple

import torch
import torch.nn as nn


def _make_conv_stack(
    input_dim: int,
    conv_channels: Tuple[int, ...],
    kernel_size: int,
    dropout: float,
) -> tuple[nn.Sequential, int]:
    layers = []
    c_in = input_dim
    for c_out in conv_channels:
        layers.extend(
            [
                nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        c_in = c_out
    return nn.Sequential(*layers), c_in


def default_chord_crnn_kwargs(num_classes: int, dropout: float = 0.4, input_dim: int = 12) -> dict:
    """Kwargs to reconstruct ChordCRNN (for saving run metadata)."""
    return {
        "num_classes": num_classes,
        "input_dim": input_dim,
        "conv_channels": (64, 128, 128),
        "kernel_size": 5,
        "lstm_hidden": 128,
        "lstm_layers": 2,
        "dropout": dropout,
    }


def default_chord_crnn_attention_kwargs(num_classes: int, dropout: float = 0.4, input_dim: int = 12) -> dict:
    """Kwargs to reconstruct ChordCRNNAttention (for saving run metadata)."""
    return {
        "num_classes": num_classes,
        "input_dim": input_dim,
        "conv_channels": (64, 128, 128),
        "kernel_size": 5,
        "lstm_hidden": 128,
        "lstm_layers": 2,
        "num_attention_heads": 8,
        "dropout": dropout,
    }


def default_chord_crnn_multiscale_kwargs(num_classes: int, dropout: float = 0.4, input_dim: int = 12) -> dict:
    """Kwargs to reconstruct ChordCRNNMultiscale (for saving run metadata)."""
    return {
        "num_classes": num_classes,
        "input_dim": input_dim,
        "multiscale_kernel_sizes": (3, 7, 15),
        "multiscale_branch_channels": 64,
        "fusion_channels": 128,
        "lstm_hidden": 128,
        "lstm_layers": 2,
        "dropout": dropout,
    }


class MultiScaleConvFront(nn.Module):
    """Parallel Conv1d with different kernel sizes over time, then 1x1 fusion."""

    def __init__(
        self,
        input_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 7, 15),
        branch_channels: int = 64,
        fusion_channels: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            if k % 2 != 1:
                raise ValueError(f"multiscale kernels must be odd for same-length conv (got k={k})")
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, branch_channels, k, padding=k // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
        fused_in = branch_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_in, fusion_channels, kernel_size=1),
            nn.BatchNorm1d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F_in, T)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        return self.fusion(x)


class ChordCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 12,
        conv_channels: Tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.conv, c_in = _make_conv_stack(input_dim, conv_channels, kernel_size, dropout)
        self.lstm = nn.LSTM(
            c_in,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.lstm(x)
        return self.head(x)  # (B, T, num_classes)


class ChordCRNNAttention(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 12,
        conv_channels: Tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.conv, c_in = _make_conv_stack(input_dim, conv_channels, kernel_size, dropout)
        self.lstm = nn.LSTM(
            c_in,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        attn_dim = 2 * lstm_hidden
        if attn_dim % num_attention_heads != 0:
            raise ValueError(
                f"2*lstm_hidden={attn_dim} must be divisible by num_attention_heads={num_attention_heads}"
            )
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(attn_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.lstm(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + self.attn_dropout(attn_out))
        return self.head(x)  # (B, T, num_classes)


class ChordCRNNMultiscale(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 12,
        multiscale_kernel_sizes: Tuple[int, ...] = (3, 7, 15),
        multiscale_branch_channels: int = 64,
        fusion_channels: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.front = MultiScaleConvFront(
            input_dim=input_dim,
            kernel_sizes=multiscale_kernel_sizes,
            branch_channels=multiscale_branch_channels,
            fusion_channels=fusion_channels,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            fusion_channels,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.front(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.lstm(x)
        return self.head(x)  # (B, T, num_classes)
