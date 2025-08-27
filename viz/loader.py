from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import yaml

def _pick(arrs: Dict[str, Any], key: str, fallback: str = None):
    if key in arrs:
        return arrs[key]
    if fallback and fallback in arrs:
        return arrs[fallback]
    return None

def load_npz_bundle_or_legacy(path: str) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=False, mmap_mode="r")
    out: Dict[str, Any] = {
        "train_ids": _pick(npz, "tt/train_ids", "train_ids"),
        "depart_station": _pick(npz, "tt/depart_station", "depart_station"),
        "arrive_station": _pick(npz, "tt/arrive_station", "arrive_station"),
        "depart_time": _pick(npz, "tt/depart_time", "depart_time"),
        "arrive_time": _pick(npz, "tt/arrive_time", "arrive_time"),
        "service": _pick(npz, "tt/service", "service"),
        "direction": _pick(npz, "tt/direction", "direction"),
        "topology": _pick(npz, "meta/topology", None),
        "station_index": _pick(npz, "meta/station_index", None),
    }
    # フォールバック：無いキーを補完
    n = len(out["depart_time"])
    if out["service"] is None:
        out["service"] = np.array(["local"] * n)
    if out["direction"] is None:
        out["direction"] = np.array(["None"] * n)
    return out

def station_order_from_config(station_yaml_path: str) -> List[str]:
    with open(station_yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [s["id"] for s in raw["stations"]]
