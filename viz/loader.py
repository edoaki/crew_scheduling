from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import yaml



def station_order_from_config(station_yaml_path: str) -> List[str]:
    with open(station_yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [s["id"] for s in raw["stations"]]
