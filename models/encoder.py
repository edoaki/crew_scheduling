from typing import Tuple, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

class PARCOEncoder(nn.Module):
    def __init__(
        self,
        time_emb : nn.Module,
        station_emb: nn.Module,
        embed_dim: int = 127,
        device: Union[torch.device, str, None] = None,
    ):
        super(PARCOEncoder, self).__init__()
        self.time_emb = time_emb
        self.station_emb = station_emb
        device = device if device is not None else torch.device("cpu")

        # カテゴリ系
        self.service_emb = nn.Embedding(2,4)
        self.direction_emb = nn.Embedding(2,4)
       
        # ブール（4種）
        self.flags_proj = nn.Linear(4,4)

        self.fuse = nn.Linear(
            76,
            embed_dim
        )
        self.norm = nn.LayerNorm(embed_dim)


    def forward(    
        self,out
    ) -> Tuple[Tensor, Tensor]:
        # Transfer to embedding space

        statics = out["statics"]
        tasks = statics["tasks"]
        pad_mask = statics["pad_masks"]["tasks"].to(torch.bool)   # True=実データ
                                          # True=PAD
        valid = pad_mask.unsqueeze(-1).float()                    # [B,T,1]

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
        e_time      = self.time_emb(t_dep) + self.time_emb(t_arr)  # [B,T,Dt]
        
        # is_* は素直にスタックして射影
        is_keys = [k for k in tasks.keys() if k.startswith("is_")]
        flags   = torch.stack([tasks[k].float().to(device) for k in is_keys], dim=-1)  # [B,T,F]
        e_flags = self.flags_proj(flags)                                               # [B,T,Df]

        # 異なる次元を結合 → 最終次元を LayerNorm の normalized_shape に合わせる
        x = torch.cat([e_service, e_direction, e_ds, e_as, e_time, e_flags], dim=-1)   # [B,T,ΣD*]
        # print("x",x.shape)

        task_emb = self.fuse(x)                              # [B,T,target_dim]
        task_emb = self.norm(task_emb)                         # normalized_shape と一致
        task_emb = task_emb * valid                            # PADを最終0化（勾配も遮断）

        return task_emb, pad_mask
    