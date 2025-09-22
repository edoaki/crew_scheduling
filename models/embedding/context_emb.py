import torch
import torch.nn as nn
from .config import ContextConfig
from models.nn.transformer import TransformerBlock 

class ContextEmbedding(nn.Module):
    def __init__(
        self, 
        time_emb: nn.Module,
        station_emb: nn.Module,
        embed_dim: int = 128,
        scale_factor: int = 10,
        num_heads: int = 8,
        num_layers: int = 3,
        
    ) :
        super().__init__()

        C = ContextConfig()
        self.scale_factor = scale_factor

        self.time_emb = time_emb
        self.station_emb = station_emb

        # on_duty: bool → 小さめ埋め込み（2値）
        self.on_duty_emb = nn.Embedding(2, C.d_on_duty)
        # 連続スカラー　単純な時間3種   をまとめて射影
        # 入力: [consec_work_time, rest_remaining, duty_minutes] の3スカラー
        self.scalar_proj = nn.Sequential(
            nn.Linear(C.d_scalar_in,C.d_scalar),
            nn.ReLU(),
            nn.Linear(C.d_scalar,C.d_scalar)
        )

        # 結合→最終埋め込みへ
        fuse_in_dim = C.crew_fuse_in
        
        self.crew_fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.crew_norm = nn.LayerNorm(embed_dim)

        # 時間スケール（分）: 相対時間・各時間の安定化用。必要に応じて調整
        self.register_buffer("t_scale_rel", torch.tensor(480.0))   # 8時間=480分を1.0に
        self.register_buffer("t_clip_rel", torch.tensor(24*60.0))  # 1日の範囲でクリップ

        self.round_embed = nn.Embedding(2, 1)

        # optional layers
        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self,cached,dyns):
        crews = dyns['crews']
        local_crew_mask = dyns["pad_masks"]["crews"].to(torch.bool) # [B,w_C]

        local_crew_ids = crews['local_crew_ids']
        
        location_station = crews['crew_location_station']
        on_duty = crews['crew_on_duty'] # bool
        ready_time = crews['crew_ready_time'] # 
        consec_work_time = crews['crew_consec_work_min']
        rest_remaining = crews['crew_rest_remaining']
        duty_minutes = crews['crew_duty_minutes']
       
        # crew 
        station_shift = torch.clamp(location_station - 1 , min=0)
        e_station = self.station_emb(station_shift)   # [B, C, 16] 
        e_on_duty = self.on_duty_emb(on_duty.long())  # [B, C, 4]
        e_ready_cyc = self.time_emb(ready_time)  # [B, C, 16]

        # --- 連続スカラー（分） ---
        # 分→スケール＆クリップを同様に適用
        cw = torch.clamp(consec_work_time, min=0.0, max=self.t_clip_rel) / self.t_scale_rel
        rr = torch.clamp(rest_remaining,   min=0.0, max=self.t_clip_rel) / self.t_scale_rel
        dm = torch.clamp(duty_minutes,     min=0.0, max=self.t_clip_rel) / self.t_scale_rel

        scalars = torch.stack([cw, rr, dm], dim=-1)          # [B, C, 3]
        e_scalar = self.scalar_proj(scalars)                         # [B, C, 12]

        # --- 結合 → 射影 → LayerNorm ---
        x = torch.cat([e_station, e_on_duty, e_ready_cyc, e_scalar], dim=-1)  # [B, C, Σ]
        h = self.crew_fuse(x)                                                # [B, C, H]
        local_crew_embed = self.crew_norm(h) # [B ,w_C ,H]
        w_C = local_crew_embed.size(1)
        local_crew_mask = dyns["pad_masks"]["crews"].to(torch.bool) # [B,w_C]
        
        tasks = dyns['tasks']
        local_task_ids = tasks['local_task_ids']
        local_task_mask = dyns["pad_masks"]["tasks"].to(torch.bool) # [B,w_T]
        
        round_bool = tasks['round_bool'] # [B,w_T]
        # print("round_bool",round_bool.shape)
        # print(round_bool)
        round_idx = round_bool.to(dtype=torch.long)
        round_emb = self.round_embed(round_idx)  # [B, w_T, 1]

        task_emb = cached.static_embeddings  # [B, A, H-1]
        local_task_emb = self._gather_tasks(task_emb, local_task_ids)       # [B, w_T ,H-1]

        local_task_emb = torch.cat([local_task_emb, round_emb], dim=-1)  # [B, w_T, H]
        
        context = torch.cat([local_crew_embed,local_task_emb],dim=1) # [B, w_C+w_T, H]
        t_c_mask = torch.cat([local_crew_mask,local_task_mask],dim=1)
        # print("context",context.shape)
        # print("t_c_mask",t_c_mask.shape)

        for layer in self.layers:
            context = layer(context, t_c_mask)

        crew_ctx = context[:, :w_C, :]   # [B, w_C, H]
        task_ctx = context[:, w_C:, :]  # [B, w_T, H]

        round_mask = local_task_mask.bool() & round_bool.bool()          # [B, w_T]
        r_T = round_mask.sum(dim=1).max().item()                         # 最大のTrue長

        round_task_ctx = task_ctx[:, :r_T, :]                            # [B, r_T, H]
        round_task_mask = round_mask[:, :r_T]                            # [B, r_T]

        return crew_ctx,round_task_ctx,local_crew_mask,round_task_mask
    

    @staticmethod
    def _gather_tasks(task_emb: torch.Tensor, local_task_ids: torch.Tensor) -> torch.Tensor:
        # task_emb: [B, A, H], local_task_ids: [B, W]
        B, A, H = task_emb.shape
        _, W = local_task_ids.shape
        idx = local_task_ids.unsqueeze(-1).expand(B, W, H)  # [B, W, H]
        gathered = torch.gather(task_emb, dim=1, index=idx) # [B, W, H]
        return gathered
    