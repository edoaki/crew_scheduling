
from typing import List, Dict, Any  
from dataclasses import dataclass
import torch
import torch.nn as nn
from rl_env.state import StaticObs, DynamicObs
from models.embedding.init_emb import StationEmbedding, TimeFourierEncoding,StaticTokenizerConfig

class CrewStaticTokenizer(nn.Module):
    def __init__(self, cfg: StaticTokenizerConfig,station_time_from_A: torch.Tensor):
        super().__init__()
        
        C = cfg

        self.station_emb = StationEmbedding(
            num_stations=C.num_stations,
            d_station_id=C.d_station_id,
            d_timepos=C.d_station_timepos,
            station_time_from_A=station_time_from_A
        )

        self.time_enc = TimeFourierEncoding(d_out=C.d_window_time, period=1440, n_harmonics=6)
        self.slot_emb = nn.Embedding(2, C.d_slot_label)  # am/pm
        self.signoff_enc = TimeFourierEncoding(d_out=C.d_signoff, period=1440, n_harmonics=4)

        in_dim = (
            (C.d_station_id + C.d_station_timepos) +   # start_station
            C.d_window_time * 2 +                      # assignable [start,end]
            C.d_slot_label +                           # slot label
            C.d_signoff                               # signoff limit
        )
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, C.d_model)

    def forward(self, static: List[StaticObs]) -> torch.Tensor:
        """
        static: dict with keys of StaticObs crew fields
        期待 shape: 各テンソル [B, L_c] or [L_c]
        出力: crew_tokens S = [B, L_c, d_model]
        """
        def ensure_3d(x):
            if x.dim() == 1:
                return x.unsqueeze(0)
            return x

        start_station = ensure_3d(static["crew_start_station"]).long()  # [B, L_c]
        start_min = ensure_3d(static["crew_assignable_start_min"]).long() % 1440
        end_min = ensure_3d(static["crew_assignable_end_min"]).long() % 1440
        slot_label = ensure_3d(static["crew_slot_label"]).long()
        signoff_lim = ensure_3d(static["crew_signoff_limit_min"]).long() % 1440

        B, Lc = start_station.shape
        num_st = self.cfg.num_stations

        e_station = self.station_emb(start_station)          # [B, Lc, d_stid+d_sttime]
        e_start = self.time_enc(start_min)                   # [B, Lc, d_window]
        e_end = self.time_enc(end_min)
        e_slot = self.slot_emb(slot_label)                   # [B, Lc, d_slot]
        e_sign = self.signoff_enc(signoff_lim)               # [B, Lc, d_signoff]

        concat = torch.cat([e_station, e_start, e_end, e_slot, e_sign], dim=-1)
        S = self.proj(self.norm(concat))  # [B, L_c, d_model]
        return S
