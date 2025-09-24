from typing import Tuple

import torch
import torch.nn as nn

from dataclasses import dataclass, fields
from typing import Tuple

from tensordict import TensorDict
from torch import Tensor


@dataclass
class PrecomputedCache:
    static_embeddings: Tensor
    static_task_mask: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))


class PARCODecoder(nn.Module):
    """
    - マルチスタートなし（num_starts 引数や batchify/unbatchify は排除）
    - use_pos_token なし
    - graph_context_cache なし（別途のキャッシュ切替機構を持たない）
    - context_embedding / dynamic_embedding / pointer は __init__ で外部から必須注入
    - pointer は通常の PointerAttention を想定（インターフェースは AM と同じ）
    """

    def __init__(
        self,
        embed_dim: int,
        context_embedding: nn.Module,
        pair_encoding: nn.Module,
        pointer: nn.Module,
        linear_bias: bool = False,
    ):
        super().__init__()
        # 必須注入の確認
        assert context_embedding is not None, "context_embedding を __init__ で渡してください"
        assert pointer is not None, "pointer を __init__ で渡してください（通常の PointerAttention）"

        self.embed_dim = embed_dim

        # 外部注入（内部で初期化しない）
        self.context_embedding = context_embedding
        self.pointer = pointer

        
        self.project_q = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )
        self.project_k = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )
        self.project_v = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )
        self.project_l = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )

        self.pair_encoding = pair_encoding

    # --- AttentionModelDecoder に相当するユーティリティ群（仕様は同様） ---

    def pre_decoder_hook(
        self,
        hidden,
        static_task_mask
    ):
        """デコード前フック（特に処理なし・そのまま返す）"""

        return PrecomputedCache(
            static_embeddings=hidden,
            static_task_mask = static_task_mask,
        )
        

    # --- メイン前向き計算（AttentionModelDecoder と同じ入出力仕様、ただし num_starts なし） ---

    def forward(
        self,
        out: TensorDict,
        cached: PrecomputedCache,
    ) -> Tuple[Tensor, Tensor]:
        """
        現在ステップのロジットとマスクを返す。

        Args:
            td: 環境の現在状態（TensorDict）
            cached: `_precompute_cache` で作った固定キャッシュ

        Returns:
            (logits, mask)
              logits: [B, N] または [B, 1, N]（pointer 実装に依存）
              mask:   [B, N]
        """
        dyns = out["dyns"]
        # print("window dyns task ",dyns["tasks"]["round_bool"].shape)
        # print("window dyns crew",dyns["crews"]["crew_on_duty"].shape)
        # print()
        # Q / KVL を計算
        crew_context,task_context,crew_mask,task_mask= self.context_embedding(cached,dyns)  
        
        glimpse_q = self.project_q(crew_context)
        glimpse_k = self.project_k(task_context)
        glimpse_v = self.project_v(task_context)
        logit_k   = self.project_l(task_context)
      

        
        # # # マスクは td 側（環境）から
        attn_mask = crew_mask[..., :, None] & task_mask[..., None, :]

        mask = out["masks"]["action_mask"]

        pointer_mask = attn_mask & mask

        # pointer_maskとmaskは同じはずだから判定 形状だけでなく、中身も全て同じはず
        if pointer_mask.shape != mask.shape or not torch.all(pointer_mask == mask):
            print("pointer_mask",pointer_mask)
            print("action_mask",mask)
            raise ValueError("pointer_mask と action_mask が異なります")
        
        pair_bias_info = out["pairs"]
        pair_bias = self.pair_encoding(pair_bias_info)
        pair_bias = pair_bias.masked_fill(attn_mask == 0, 0.0)  
        # print("pair_bias",pair_bias)
        # # PointerAttention に渡してロジットを得る
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k,pointer_mask,pair_bias)
        # print("decoder logits",logits)
        return logits,pointer_mask
  