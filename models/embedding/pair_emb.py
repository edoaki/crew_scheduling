import torch
import torch.nn as nn

class PairMlp(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(5)  # 特徴数=5 を正規化（位置ごとに特徴軸で正規化）
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self,pair_bias_info):  # feats: [B,C,T,5]
        # print("pair_bias_info in pair emb",pair_bias_info["is_hitch"].shape)
        feats = self.preprocess_pair_features(pair_bias_info)  # [B,C,T,5]
        x = self.ln(feats)
        pair_bias = self.mlp(x)  # [B,C,T,out_dim]
        pair_bias = pair_bias.squeeze(-1)  # [B,C,T] (out_dim=1 の場合)
        return pair_bias
    

    def preprocess_pair_features(self,pair_bias_info,
                                minute_cap: int =120,   # 例: 2時間
                                rel_cap: int = 120,      # 例: 6時間 
                                use_log: bool = False):
        # 入力は全て [B,C,T]
        is_hitch = pair_bias_info["is_hitch"].to(torch.float32)
        hitch_minutes = pair_bias_info["hitch_minutes"].clamp_min(0).to(torch.float32)
        is_continuous = pair_bias_info["is_continuous"].to(torch.float32)
        rel_time = pair_bias_info["rel_time"].clamp_min(0).to(torch.float32)  # 0埋め済み
        on_duty = pair_bias_info["on_duty"].to(torch.float32)

        if use_log:
            # log1pでダイナミックレンジを圧縮 → 代表的な上限で割って[0,1]目安に
            hitch = torch.log1p(hitch_minutes)
            rel = torch.log1p(rel_time)
            hitch = hitch / torch.log1p(torch.tensor(float(minute_cap), device=hitch_minutes.device))
            rel   = rel   / torch.log1p(torch.tensor(float(rel_cap), device=rel_time.device))
        else:
            # 線形スケーリング
            hitch = hitch_minutes / float(minute_cap)
            rel   = rel_time      / float(rel_cap)

        hitch = hitch.clamp_(0.0, 1.0)
        rel   = rel.clamp_(0.0, 1.0)

        # [B,C,T,5] にまとめる（並びは任意だが固定する）
        feats = torch.stack([is_hitch, hitch, is_continuous, rel, on_duty], dim=-1)
        return feats  # [B,C,T,5]
