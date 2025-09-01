from utils.io_npz import load_timetable_bundle

from timetable.generator import generate_and_save
from pathlib import Path
import numpy as np

DATA_DIR = Path("data")
CONFIG_DIR = Path("config")

station_yaml = str(CONFIG_DIR / "station.yaml")
train_yaml = str(CONFIG_DIR / "train.yaml")
constraints_yaml = str(CONFIG_DIR / "constraints.yaml")
out_path = str(DATA_DIR / f"timetable.npz")
ok, msg = generate_and_save(station_yaml, train_yaml,constraints_yaml,out_path)

