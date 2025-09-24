from utils.yaml_loader import load_yaml
from rl_env.generator import CrewARGenerator 
from rl_env.batch_env import VecCrewAREnv
from models.embedding.common_emb import StationEmbedding, TimeFourierEncoding
from models.embedding.context_emb import ContextEmbedding
from models.pointer_attention import PointerAttention
from models.embedding.pair_emb import PairMlp
from models.embedding.utils import load_station_time_from_A
from models.encoder import PARCOEncoder
from models.decoder import PARCODecoder
from models.policy import Policy
from rl_env.reward import calculate_reward


import os
import copy
import torch
from torch.nn.utils import clip_grad_norm_


from pathlib import Path

DATA_DIR = Path("data")
CONFIG_DIR = Path("test2_config")

station_yaml = str(CONFIG_DIR / "station.yaml")
train_yaml = str(CONFIG_DIR / "train.yaml")
constraints_yaml = str(CONFIG_DIR / "constraints.yaml")
crew_yaml = str(CONFIG_DIR / "crew.yaml")
encoding_yaml = str(CONFIG_DIR / "encoding.yaml")

data_path = DATA_DIR / "sample.npz"

generator= CrewARGenerator(station_yaml=station_yaml,
                            train_yaml=train_yaml,
                            constraints_yaml=constraints_yaml,
                            crew_yaml=crew_yaml
                            )

constraints = load_yaml(constraints_yaml)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

station_time_from_A = load_station_time_from_A(station_yaml,encoding_yaml)

station_emb = StationEmbedding(
            num_stations=6,
            d_station_id=8,
            d_timepos=16,
            station_time_from_A=station_time_from_A
        )
time_emb = TimeFourierEncoding(d_out=16, period=1440, n_harmonics=8)
crew_emb = Crew_DyamicEmbedding = None

encoder = PARCOEncoder(time_emb=time_emb,
                        station_emb=station_emb,
                        embed_dim=127,
                       )

context_emb = ContextEmbedding(
    time_emb=time_emb,
    station_emb=station_emb,
    embed_dim=128,
    scale_factor=10,
)

pair_encoding = PairMlp(hidden_dim=16,out_dim=1)
pointer = PointerAttention(embed_dim=128,num_heads=8)

decoder = PARCODecoder(context_embedding=context_emb,
                       pair_encoding= pair_encoding,
                        pointer=pointer,
                          embed_dim=128,
                       )

batch_size = 100
vec_env = VecCrewAREnv(generator=generator,constraints=constraints,batch_size=batch_size,device=device) 
policy = Policy(encoder=encoder,decoder=decoder) 

# 設定
num_epochs = 100
updates_per_epoch = 10
lr = 1e-4
grad_clip = 1.0
improve_patience = 10  # 10エポック改善なしで早期終了
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# オプティマイザ
optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

# ベースラインモデル（初期は同一重みのコピー）
baseline_policy = copy.deepcopy(policy)

# 進捗トラッキング
best_baseline_mean_reward = None
epochs_since_improve = 0
print("Starting training...")
for epoch in range(1, num_epochs + 1):
    policy.train()
    for _ in range(updates_per_epoch):
        # 同じ問題を毎回生成（今回は生成結果が同一想定）
        td_batch = vec_env.generate_batch_td(B=batch_size)

        env_out = vec_env.reset(td_batch)

        # 方策サンプリング（train：確率的）
        outdict = policy(env_out=env_out, vec_env=vec_env, phase="train")
        sol = outdict["solution"]
        logprobs = outdict["log_likelihood"]  # [B]
        # 報酬（reward = -cost で既に計算済み想定）
        reward = calculate_reward(sol, vec_env, device=device)  # [B]
        
        # baseline（同じ env_out 上で、通常 greedy 推論）
        with torch.no_grad():
            base_env_out = vec_env.reset(td_batch)
            base_out = baseline_policy(env_out=base_env_out, vec_env=vec_env, phase="val")
            base_sol = base_out["solution"]
            baseline_reward = calculate_reward(base_sol, vec_env, device=device)  # [B]
        # print("reward :",reward,"/ base :", baseline_reward)
        # REINFORCE 損失： -E[(R - b) * logpi]
        advantage = (reward - baseline_reward).detach()
        adv_mean = advantage.mean()
        adv_std = advantage.std(unbiased=False).clamp_min(1e-8)
        advantage = (advantage - adv_mean) / adv_std  # 標準化
        
        loss = -(advantage * logprobs).mean()
        # print(f"[Epoch {epoch:03d}] loss={loss.item():.6f}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

    # ====== エポック評価（同一生成を都度実行、評価もgreedy） ======
    policy.eval()
    with torch.no_grad():
        eval_td = vec_env.generate_batch_td(B=batch_size)
        eval_out = vec_env.reset(eval_td)
    
        cur_out = policy(env_out=eval_out, vec_env=vec_env, phase="val")

        cur_sol = cur_out["solution"]
        cur_reward = calculate_reward(cur_sol, vec_env, device=device)  # [B]
        cur_mean_reward = cur_reward.mean().item()

        base_eval_out = vec_env.reset(eval_td)
        base_out = baseline_policy(env_out=base_eval_out, vec_env=vec_env, phase="val")
        base_sol = base_out["solution"]
        baseline_reward = calculate_reward(base_sol, vec_env, device=device)  # [B]
        base_mean_reward = baseline_reward.mean().item()
        
    improved = (best_baseline_mean_reward is None) or (cur_mean_reward > best_baseline_mean_reward)

    if improved:
        # ベースライン更新＆保存
        baseline_policy = copy.deepcopy(policy).to(device)
        best_baseline_mean_reward = cur_mean_reward
        epochs_since_improve = 0

        ckpt_path = os.path.join(save_dir, f"{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "policy_state_dict": policy.state_dict(),
            "baseline_state_dict": baseline_policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cur_mean_reward": cur_mean_reward,
            "base_mean_reward": base_mean_reward,
        }, ckpt_path)
    else:
        epochs_since_improve += 1

    print(f"[Epoch {epoch:03d}] "
        f"train_loss={loss.item():.6f}  "
        f"cur_mean_reward={cur_mean_reward:.6f}  "
        f"baseline_mean_reward={base_mean_reward:.6f}  "
        f"improved={improved}  "
        f"no_improve_for={epochs_since_improve}")

    # 10エポック連続で改善なしなら早期終了
    if epochs_since_improve >= improve_patience:
        print(f"Early stop at epoch {epoch} (no improvement for {improve_patience} epochs).")
        break