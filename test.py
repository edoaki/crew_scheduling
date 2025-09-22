from utils.yaml_loader import load_yaml

from env.generator import CrewARGenerator ,save_npz,load_npz
from env.env import CrewAREnv
from pathlib import Path
import numpy as np

DATA_DIR = Path("data")
CONFIG_DIR = Path("test_config")

station_yaml = str(CONFIG_DIR / "station.yaml")
train_yaml = str(CONFIG_DIR / "train.yaml")
constraints_yaml = str(CONFIG_DIR / "constraints.yaml")
crew_yaml = str(CONFIG_DIR / "crew.yaml")

data_path = DATA_DIR / "sample.npz"

generator = CrewARGenerator(station_yaml=station_yaml,
                            train_yaml=train_yaml,
                            constraints_yaml=constraints_yaml,
                            crew_yaml=crew_yaml
                            )

from models.policy import DummyModel
from models.agent_handler import GreedyAgentHandler

model = DummyModel()
agent_handler = GreedyAgentHandler()

td = generator.generate()
save_npz(data_path, td)

td = load_npz(data_path)

constraints = load_yaml(constraints_yaml)
env = CrewAREnv(constraints)


static_obs,dyn_obs,mask ,pair_info= env.reset(td)

# # model.reset(static_obs)
# print(env.dyn.crew_station)
# print(env.dyn.crew_ready_time)

# print(mask)

# action_prob = model(dyn_obs)
# assignment = agent_handler.assign(mask, action_prob)
# print(assignment)

# dyn_obs,mask,pair_info,reward, done, info = env.step(assignment)
# print("after step")
# print(env.dyn.task_assign)
# print(env.dyn.train_last_crew_id)
# print(env.dyn.crew_station)
# print(env.dyn.crew_ready_time


# ===== rollout (1 episode) =====
total_reward = 0.0
step_idx = 0
history = []  # 必要ならログ用
done = False
print("round ",env.static.num_rounds)
while not done:
    # 1) 方策から行動確率（あるいはスコア）を計算
    action_prob = model(dyn_obs)

    # 2) マスクを考慮して割当を決定（貪欲ハンドラ）
    assignment = agent_handler.assign(mask, action_prob)
    # print(assignment)
    # print(mask)
    # 3) 環境を1ステップ進める
    dyn_obs, mask, pair_info, reward, done, info = env.step(assignment)

    # print("ready ",dyn_obs.crew_ready_time)

    # 4) 報酬を集計（torch/npのスカラー両対応）
    r = reward.item() if hasattr(reward, "item") else float(reward)
    total_reward += r

    # 5) ログ（必要なら）
    history.append({
        "step": step_idx,
        "reward": r,
        "done": done,
        "info": info,
        "assignment": assignment,
    })

    step_idx += 1
    print(f"[Step] step={step_idx}, reward={r}, done={done}, info={info} round_time = {env.dyn.now_round_time}")
    print()

print("--Final task_assign")
print("train_id ",env.static.train_id)  
print("dep station ",env.static.depart_station)
print("assign ",env.dyn.task_assign)
print(f"[Episode] steps={step_idx}, total_reward={total_reward}")
