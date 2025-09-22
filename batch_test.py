
from utils.yaml_loader import load_yaml

from rl_env.generator import CrewARGenerator ,save_npz,load_npz
from rl_env.batch_env import VecCrewAREnv
from rl_env.env import CrewAREnv
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

B = 1  # バッチサイズ

def batch_generete(generator, B):
    td_batch = [generator.generate() for _ in range(B)]
    return td_batch

td_batch = batch_generete(generator,B=1)

constraints = load_yaml(constraints_yaml)

env = VecCrewAREnv(CrewAREnv,generator, constraints, batch_size=B)

# modelの初期化
from models.model import ParcoModel
model = ParcoModel(


)

# static をモデルで前処理
model(env,td_batch)

# # 単体用 assign を流用（バッチの薄いラッパ）
# assignments = [agent_handler.assign(m, p) for m, p in zip(mask_batch, action_prob_batch)]

# dyn_obs_batch, mask_batch, pair_info_batch, reward, done, infos = env.step(assignments)


