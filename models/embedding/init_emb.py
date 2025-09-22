import math
from typing import Dict, Optional
import torch
from .config import StaticInitConfig
from .common_emb import StationEmbedding, TimeFourierEncoding
import torch.nn as nn

class InitEnbeddings(nn.Module):
    """
    static から Task/Crew のトークン T, S を作る“前処理＋埋め込み”層。
     
    まずはタスクのみで設計

    """
    def __init__(self, station_time_from_A: torch.Tensor,
                device: Optional[torch.device] = None):
        super().__init__()
        C = StaticInitConfig()
        
        self.d_model = C.d_model

        device = device if device is not None else torch.device("cpu")

        # カテゴリ系
        self.service_emb = nn.Embedding(C.num_services, C.d_service)
        self.direction_emb = nn.Embedding(C.num_directions, C.d_direction)
        
        # 駅と時間
        self.station_emb = StationEmbedding(
            num_stations=C.num_stations,
            d_station_id=C.d_station_id,
            d_timepos=C.d_station_timepos,
            station_time_from_A=station_time_from_A
        )
        self.time_enc = TimeFourierEncoding(d_out=C.d_timepos, period=1440, n_harmonics=8)
       
        # ブール（4種）
        self.flags_proj = nn.Linear(4, C.d_flags)

        self.fuse = nn.Linear(
            C.d_service + C.d_direction + 
            2 * (C.d_station_id + C.d_station_timepos) +
            C.d_timepos + C.d_flags,
            C.d_model
        )
        self.norm = nn.LayerNorm(C.d_model)
                                                            
    def forward(self, statics):
        tasks = statics["tasks"]
        real_mask = statics["pad_masks"]["tasks"].to(torch.bool)   # True=実データ
        attn_mask = real_mask                                     # True=PAD
        valid = real_mask.unsqueeze(-1).float()                    # [B,T,1]

        device = next(self.parameters()).device
        service   = tasks["service"].long().to(device)             # +1 済み (0=PAD, 1..K)
        direction = tasks["direction"].long().to(device)           # +1 済み
        ds        = tasks["depart_station"].long().to(device)      # +1 済み (1..6)
        as_       = tasks["arrive_station"].long().to(device)      # +1 済み (1..6)
        t_dep     = tasks["depart_time"].float().to(device)        # 分
        t_arr     = tasks["arrive_time"].float().to(device)        # 分

        # カテゴリは有効位置だけ -1 して 0..K-1 に合わせる（PADは後段で0化）
        service_shift   = torch.clamp(service - 1, min=0)
        direction_shift = torch.clamp(direction - 1, min=0)
        ds_shift        = torch.clamp(ds - 1, min=0)               # 1..6 → 0..5
        as_shift        = torch.clamp(as_ - 1, min=0)
       
        # 埋め込み／エンコーディング
        e_service   = self.service_emb(service_shift)              # [B,T,Ds]
        e_direction = self.direction_emb(direction_shift)          # [B,T,Dd]
        e_ds        = self.station_emb(ds_shift)                   # [B,T,De]
        e_as        = self.station_emb(as_shift)                   # [B,T,De]
        e_time      = self.time_enc(t_dep) + self.time_enc(t_arr)  # [B,T,Dt]

        # is_* は素直にスタックして射影
        is_keys = [k for k in tasks.keys() if k.startswith("is_")]
        flags   = torch.stack([tasks[k].float().to(device) for k in is_keys], dim=-1)  # [B,T,F]
        e_flags = self.flags_proj(flags)                                               # [B,T,Df]

        # 異なる次元を結合 → 最終次元を LayerNorm の normalized_shape に合わせる
        x = torch.cat([e_service, e_direction, e_ds, e_as, e_time, e_flags], dim=-1)   # [B,T,ΣD*]

        task_emb = self.fuse(x)                              # [B,T,target_dim]
        task_emb = self.norm(task_emb)                         # normalized_shape と一致
        task_emb = task_emb * valid                            # PADを最終0化（勾配も遮断）

        return task_emb, attn_mask
