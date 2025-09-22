
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
encoding_yaml = str(CONFIG_DIR / "encoding.yaml")

data_path = DATA_DIR / "sample.npz"

generator1= CrewARGenerator(station_yaml=station_yaml,
                            train_yaml=train_yaml,
                            constraints_yaml=constraints_yaml,
                            crew_yaml=crew_yaml
                            )


CONFIG2_DIR = Path("test2_config")
station2_ymal = str(CONFIG2_DIR / "station.yaml")
train2_yaml = str(CONFIG2_DIR / "train.yaml")
constraints2_yaml = str(CONFIG2_DIR / "constraints.yaml")
crew2_yaml = str(CONFIG2_DIR / "crew.yaml")
encoding2_yaml = str(CONFIG2_DIR / "encoding.yaml")

generator2 = CrewARGenerator(station_yaml=station2_ymal,
                            train_yaml=train2_yaml,
                            constraints_yaml=constraints2_yaml,
                            crew_yaml=crew2_yaml
                            )

from models.embedding.utils import load_station_time_from_A
from models.encoder import PARCOEncoder


from rl_env.batch_env import VecCrewAREnv
constraints = load_yaml(constraints_yaml)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


B = 2
vec_env = VecCrewAREnv(generator=generator1,constraints=constraints,batch_size=B,device=device)

td1 = generator1.generate()
td2 = generator2.generate()

td_batch = [td1,td2]
print("batch 0 task ",td_batch[0]["depart_time"].shape)
print("batch 1 task ",td_batch[1]["depart_time"].shape)

print("batch 0 crew ",td_batch[0]["start_station_idx"].shape)
print("batch 1 crew ",td_batch[1]["start_station_idx"].shape)

out = vec_env.reset(td_batch)

print("static task shape",out['statics']["tasks"]["service"].shape)


print()
from models.embedding.common_emb import StationEmbedding, TimeFourierEncoding
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

hidden,static_task_mask = encoder(out)

from models.embedding.context_emb import ContextEmbedding
context_emb = ContextEmbedding(
    time_emb=time_emb,
    station_emb=station_emb,
    embed_dim=128,
    scale_factor=10,
)

from models.pointer_attention import PointerAttention
from models.embedding.pair_emb import PairMlp

pair_encoding = PairMlp(hidden_dim=16,out_dim=1)
pointer = PointerAttention(embed_dim=128,num_heads=8)

from models.decoder import PARCODecoder
decoder = PARCODecoder(context_embedding=context_emb,
                       pair_encoding= pair_encoding,
                        pointer=pointer,
                          embed_dim=128,
                       )

from models.policy import Policy
policy = Policy(encoder=encoder,decoder=decoder)

outdict = policy(env_out=out,vec_env=vec_env,phase="train")

