from pathlib import Path
import torch
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

def load_configs(config_dir: Path):
    station_yaml = str(config_dir / "station.yaml")
    train_yaml = str(config_dir / "train.yaml")
    constraints_yaml = str(config_dir / "constraints.yaml")
    crew_yaml = str(config_dir / "crew.yaml")
    encoding_yaml = str(config_dir / "encoding.yaml")
    constraints = load_yaml(constraints_yaml)
    return {
        "paths": {
            "station_yaml": station_yaml,
            "train_yaml": train_yaml,
            "constraints_yaml": constraints_yaml,
            "crew_yaml": crew_yaml,
            "encoding_yaml": encoding_yaml,
        },
        "constraints": constraints,
    }

def build_generator(paths: dict):
    return CrewARGenerator(
        station_yaml=paths["station_yaml"],
        train_yaml=paths["train_yaml"],
        constraints_yaml=paths["constraints_yaml"],
        crew_yaml=paths["crew_yaml"],
    )

def build_env(generator, constraints: dict, batch_size: int, device: torch.device):
    return VecCrewAREnv(generator=generator, constraints=constraints, batch_size=batch_size, device=device)

def build_embeddings(station_yaml: str, encoding_yaml: str):
    station_time_from_A = load_station_time_from_A(station_yaml, encoding_yaml)
    station_emb = StationEmbedding(
        num_stations=6,
        d_station_id=8,
        d_timepos=16,
        station_time_from_A=station_time_from_A
    )
    time_emb = TimeFourierEncoding(d_out=16, period=1440, n_harmonics=8)
    return station_emb, time_emb

def build_policy(station_emb, time_emb):
    encoder = PARCOEncoder(
        time_emb=time_emb,
        station_emb=station_emb,
        embed_dim=127,
    )
    context_emb = ContextEmbedding(
        time_emb=time_emb,
        station_emb=station_emb,
        embed_dim=128,
        scale_factor=10,
    )
    pair_encoding = PairMlp(hidden_dim=16, out_dim=1)
    pointer = PointerAttention(embed_dim=128, num_heads=8)
    decoder = PARCODecoder(
        context_embedding=context_emb,
        pair_encoding=pair_encoding,
        pointer=pointer,
        embed_dim=128,
    )
    policy = Policy(encoder=encoder, decoder=decoder)
    return policy
