import math
from typing import Dict, Optional
import torch
import torch.nn as nn


# ---- ユーティリティ：時間の sin/cos 符号化（複数高調波） ----
class TimeFourierEncoding(nn.Module):
    def __init__(self, d_out: int, period: int = 1440, n_harmonics: int = 8):
        super().__init__()
        assert d_out % 2 == 0, "d_out は偶数にしてください（sin/cos の対にするため）"
        self.period = period
        self.n_harmonics = n_harmonics
        # 周波数セット（1,2,4,...）を用意
        k = torch.arange(n_harmonics).float()
        self.register_buffer("freq", 2.0 ** k * (2.0 * math.pi / period))  # [H]

        # 出力を d_out に合わせるための線形（H*2 → d_out）
        self.proj = nn.Linear(n_harmonics * 2, d_out)

    def forward(self, minutes: torch.Tensor) -> torch.Tensor:
        # minutes: [...], 実数でも整数でも可
        x = minutes.float()[..., None] * self.freq  # [..., H]
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [..., H*2]
        return self.proj(enc)  # [..., d_out]


# ---- 駅の埋め込み：駅IDの学習埋め込み + Aからの移動時間の周期埋め込み ----
class StationEmbedding(nn.Module):
    def __init__(self, num_stations: int, d_station_id: int, d_timepos: int,
                 station_time_from_A: torch.Tensor):
        """
        station_time_from_A: [num_stations] (分) を事前計算して渡す
        """
        super().__init__()
        self.id_emb = nn.Embedding(num_stations, d_station_id)
        self.register_buffer("t_from_A", station_time_from_A.float())
        self.t_scale = float(station_time_from_A.max()) + 1e-6
        self.t_proj = nn.Sequential(
            nn.Linear(1, d_timepos),
            nn.ReLU(),
            nn.Linear(d_timepos, d_timepos)
)

    def forward(self, station_ids: torch.Tensor) -> torch.Tensor:
        e_id = self.id_emb(station_ids)
        tA = self.t_from_A[station_ids] / self.t_scale
        e_t = self.t_proj(tA.unsqueeze(-1))
        return torch.cat([e_id, e_t], dim=-1)
