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

td_batch = vec_env.generate_batch_td(B=batch_size)
env_out = vec_env.reset(td_batch)

# 方策サンプリング（train：確率的）
outdict = policy(env_out=env_out, vec_env=vec_env, phase="train")
sol = outdict["solution"]
# print("station ",env_out["statics"]["tasks"]["depart_station"])
# print("solution:", sol)
logprobs = outdict["log_likelihood"]  # [B]
# 報酬（reward = -cost で既に計算済み想定）
reward = calculate_reward(sol, vec_env, device=device)  # [B]

from rl_env.reward import evaluate_solution
batch_totals = []
for i in range(batch_size):
    env = vec_env.envs[i]
    s = env.static
    T = s.num_tasks
    C = s.num_crews

    # print(f"=== Batch {i} ===")

    # sol[i] は [T]
    sol_i = sol[i]
    # print("sol ",sol_i)
    # print("dep ",s.depart_station)

    result = evaluate_solution(s, sol_i)
    batch_totals.append({
        "total_work_time": result["total_work_time"],
        "total_hitch_time": result["total_hitch_time"],
        "unassigned_count": result["unassigned_count"],
    })

# unassigned_countの分布を見る
from collections import Counter
uc_list = [bt["unassigned_count"] for bt in batch_totals]
print("unassigned_count",Counter(uc_list))