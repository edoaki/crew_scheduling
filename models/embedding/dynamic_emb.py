import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from typing import Tuple


class DynamicEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.post_cat_project = nn.Linear(self.embed_dim , self.embed_dim, bias=False)


    @staticmethod
    def _gather_tasks(task_emb: torch.Tensor, local_task_ids: torch.Tensor) -> torch.Tensor:
        # task_emb: [B, A, H], local_task_ids: [B, W]
        B, A, H = task_emb.shape
        _, W = local_task_ids.shape
        idx = local_task_ids.unsqueeze(-1).expand(B, W, H)  # [B, W, H]
        gathered = torch.gather(task_emb, dim=1, index=idx) # [B, W, H]
        return gathered

    def forward(
        self,
        cache: TensorDict,
        dyns: TensorDict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task_emb = cache.static_embeddings  # [B, A, H]
        local_task_ids = dyns['tasks']['local_task_ids']
        task_mask = dyns["pad_masks"]["tasks"]
        round_bool = dyns['tasks']['round_bool']

        device = task_emb.device
        dtype  = task_emb.dtype
        
        # 1) 局所タスク埋め込みを取り出し
        e_win = self._gather_tasks(task_emb, local_task_ids)          # [B, W, H]
        
        # 2) パディング位置をゼロ化（id=0が実タスクでも、maskで区別する）
        m = task_mask.unsqueeze(-1).to(dtype)                          # [B, W, 1]
        e_win = e_win * m                                              # [B, W, H]

        # 3) round_bool を [B,W,1] float に
        r = round_bool.to(dtype).unsqueeze(-1)                         # [B, W, 1]

        # 4) 特徴統合（concatがシンプルで強い）
        z = torch.cat([e_win, r], dim=-1)                              # [B, W, H+1]
        z = self.post_cat_project(z)                                   # [B, W, H]s
        return z, task_mask
    