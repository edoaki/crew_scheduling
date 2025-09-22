# context_emb.py
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.nn.transformer import TransformerBlock as CommunicationLayer
from models.embedding.init_emb import StationEmbedding, TimeFourierEncoding


@dataclass
class CtxDims:
    # 出力埋め込み
    embed: int = 256

    # 個別モジュールの次元
    station: int = 64
    time: int = 64
    round: int = 32
    task_cache: int = 128  # タスク埋め込みキャッシュの1要素次元
    # 必要なら他の素片もここに追加（例：flagsやdirectionなど）

    # CommunicationLayer 関連（使うなら）
    comm_hidden: int = 256
    comm_heads: int = 8
    comm_layers: int = 2

    @property
    def local_task_in(self) -> int:
        # local_task_from_cache [*, task_cache] と round_embed [*, round] の結合
        return self.task_cache + self.round

    @property
    def fuse_indim(self) -> int:
        # 「C.○ の足し算だけ」で決めたいとの要望に合わせて、足し算のみで定義
        # ここでは例として station と time を fuse する想定（必要に応じて項目を足してください）
        return self.station + self.time


class ContextEmbedding(nn.Module):
    """
    次元管理を dataclass(CtxDims) に集約した Context Embedding 実装。
    - すべての Linear 入出力は C.○ を参照（ハードコーディングの数値は排除）
    - fuse_indim は C.○ の足し算のみで定義
    - init_emb（StationEmbedding, TimeFourierEncoding）の設計方針に寄せた構成
    """
    def __init__(
        self,
        embed_dim: int = 256,
        scale_factor: int = 10,
        use_comm_layer: bool = True,
        station_time_from_A: Optional[torch.Tensor] = None,
        dims: Optional[CtxDims] = None,
    ) -> None:
        super().__init__()

        # 次元定義：引数 embed_dim を優先して C に反映
        if dims is None:
            dims = CtxDims(embed=embed_dim)
        else:
            # 矛盾回避：明示 embed_dim が与えられたら C にも反映
            dims.embed = embed_dim
        self.C = dims

        # ---- 基本エンコーダ ----
        # 駅（場所）埋め込み
        self.station_emb = StationEmbedding(d_out=self.C.station, scale_factor=scale_factor)

        # 時刻のフーリエ埋め込み（偶数次元は TimeFourierEncoding 内で担保される想定）
        self.time_enc = TimeFourierEncoding(d_out=self.C.time)

        # ---- fuse：station と time を結合して embed へ写像（必要な素片を追加するなら fuse_indim にも C.* を足す）----
        self.fuse = nn.Linear(self.C.fuse_indim, self.C.embed)

        # ---- Local Task 用：cache + round を結合して embed へ ----
        self.local_task_proj = nn.Linear(self.C.local_task_in, self.C.embed)

        # ---- Communication Layer（任意）----
        self.use_comm_layer = use_comm_layer
        if self.use_comm_layer:
            self.comm = nn.ModuleList([
                CommunicationLayer(
                    d_model=self.C.comm_hidden,
                    nhead=self.C.comm_heads,
                )
                for _ in range(self.C.comm_layers)
            ])
            # 入出力整形（embed <-> comm_hidden）
            self.to_comm = nn.Linear(self.C.embed, self.C.comm_hidden)
            self.from_comm = nn.Linear(self.C.comm_hidden, self.C.embed)

        # 参考として保持（使わないなら無視される）
        self.station_time_from_A = station_time_from_A

        # 正規化
        self.norm = nn.LayerNorm(self.C.embed)

    # ---------------------------------------------------------------------
    # 公開API
    # ---------------------------------------------------------------------
    def forward(
        self,
        *,
        station_ids: torch.Tensor,
        minutes: torch.Tensor,
        task_embed_cache: Optional[torch.Tensor] = None,
        local_tasks: Optional[torch.Tensor] = None,
        round_embed: Optional[torch.Tensor] = None,
        apply_comm: bool = False,
    ) -> torch.Tensor:
        """
        汎用 forward。
        - 最低限、station_ids と minutes から文脈埋め込みを作る
        - （任意）local task 情報が来たら cache+round を用いたローカルタスク埋め込みを fuse に混ぜる
        形状:
          station_ids: [B, L] など（StationEmbedding の仕様に追従）
          minutes:     [B, L] など（TimeFourierEncoding の仕様に追従）
          task_embed_cache: [B, N_task, C.task_cache]
          local_tasks:      [B, L]  （各位置のローカルID: 0..N_task-1）
          round_embed:      [B, L, C.round]
        戻り:
          context_emb: [B, L, C.embed]
        """

        # 基本素片の作成
        e_station = self.station_emb(station_ids)     # [B, L, C.station]
        e_time = self.time_enc(minutes)               # [B, L, C.time]

        x = torch.cat([e_station, e_time], dim=-1)    # [B, L, C.fuse_indim]
        context = self.fuse(x)                        # [B, L, C.embed]

        # もし local task 情報があれば融合（concat -> Linear 投入にせず、ここでは単純和にする等は要件に応じて変更可）
        if (task_embed_cache is not None) and (local_tasks is not None) and (round_embed is not None):
            local_task_emb = self.embed_local_task(task_embed_cache, local_tasks, round_embed)  # [B, L, C.embed]
            context = context + local_task_emb

        context = self.norm(context)

        if apply_comm and self.use_comm_layer:
            # TransformerBlock は [L, B, D] 前提のことが多いので転置
            h = self.to_comm(context)                 # [B, L, C.comm_hidden]
            h = h.transpose(0, 1).contiguous()        # [L, B, C.comm_hidden]
            for blk in self.comm:
                h = blk(h)                            # [L, B, C.comm_hidden]
            h = h.transpose(0, 1).contiguous()        # [B, L, C.comm_hidden]
            context = self.from_comm(h)               # [B, L, C.embed]
            context = self.norm(context)

        return context

    @torch.no_grad()
    def spec(self) -> CtxDims:
        """次元仕様を返す（デバッグ／設定確認用）"""
        return self.C

    # ---------------------------------------------------------------------
    # 内部ユーティリティ
    # ---------------------------------------------------------------------
    def embed_local_task(
        self,
        task_embed_cache: torch.Tensor,  # [B, N_task, C.task_cache]
        local_tasks: torch.Tensor,       # [B, L]
        round_embed: torch.Tensor,       # [B, L, C.round]
    ) -> torch.Tensor:
        """
        ローカルIDで指定されたタスク埋め込みとラウンド埋め込みを結合 → C.embed へ。
        ここでも「nn.Linear(3, 12)」のようなリテラルは使わず、self.C.* のみ参照。
        """
        B, L = local_tasks.shape
        _, N_task, D_task = task_embed_cache.shape  # ここは入力テンソルの shape から得るが、設計上 D_task == self.C.task_cache を想定

        # [B, L, D_task] gather
        idx = local_tasks.unsqueeze(-1).expand(-1, -1, D_task)          # [B, L, D_task]
        local_task_from_cache = torch.gather(task_embed_cache, 1, idx)  # [B, L, D_task]

        # cache と round の結合 → C.local_task_in (= C.task_cache + C.round)
        cat = torch.cat([local_task_from_cache, round_embed], dim=-1)   # [B, L, C.local_task_in]

        # embed へ線形射影
        local_task_embed = self.local_task_proj(cat)                    # [B, L, C.embed]
        return local_task_embed
