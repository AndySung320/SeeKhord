"""
Frame-level chord classifier: 1D CNN over time + BiLSTM + per-frame logits.
Input: (batch, time, 12) chroma features from preprocess.
"""
from typing import Tuple

import torch
import torch.nn as nn


class ChordCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 12,
        conv_channels: Tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
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
        self.conv = nn.Sequential(*layers)
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
        b, t, f = x.shape
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.lstm(x)
        return self.head(x)  # (B, T, num_classes)
