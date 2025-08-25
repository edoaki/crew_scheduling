# utils/npz.py
import os
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import yaml


def load_npz_any(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as L:
        def pick(name: str, fallback: Optional[str] = None):
            if name in L:
                return L[name]
            if fallback and fallback in L:
                return L[fallback]
            return None

        bundle = {
            "train_ids": pick("tt/train_ids", "train_ids"),
            "depart_station": pick("tt/depart_station", "depart_station"),
            "arrive_station": pick("tt/arrive_station", "arrive_station"),
            "depart_time": pick("tt/depart_time", "depart_time"),
            "arrive_time": pick("tt/arrive_time", "arrive_time"),
            "service": pick("tt/service", "service"),
            "direction": pick("tt/direction", "direction"),
            "topology": pick("meta/topology", None),
            "station_index": pick("meta/station_index", None),
            "meta_json": pick("meta/json", None),
        }

    meta = {}
    if bundle["meta_json"] is not None:
        try:
            b = bytes(bundle["meta_json"])
            meta = json.loads(b.decode("utf-8"))
        except Exception:
            meta = {}
    bundle["meta"] = meta

    for k in ["train_ids", "depart_station", "arrive_station", "service", "direction", "topology"]:
        if bundle.get(k) is not None:
            bundle[k] = bundle[k].astype(str)

    for k in ["depart_time", "arrive_time", "station_index"]:
        if bundle.get(k) is not None:
            bundle[k] = bundle[k].astype(int)

    return bundle


def load_station_topology_and_local_times(station_yaml_path: str) -> Tuple[List[str], Dict[Tuple[str, str], int]]:
    with open(station_yaml_path, "r", encoding="utf-8") as f:
        st = yaml.safe_load(f)

    topo = list(st.get("topology", {}).get("ordered_stations", []))
    local_segs: Dict[Tuple[str, str], int] = {}

    segs = st.get("segments", {})
    def norm_key(k): return str(k).strip().lstrip("-").strip().lower()

    def parse_minutes(x) -> int:
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(round(x))
        if isinstance(x, str):
            s = x.strip()
            if ":" in s:
                h, m = s.split(":", 1)
                return int(h) * 60 + int(m)
            return int(s)
        raise ValueError(f"Unsupported time format: {x!r}")

    def ingest(key, arr):
        if not isinstance(arr, list):
            return
        if key == "local":
            for seg in arr:
                u, v = seg["from"], seg["to"]
                t = parse_minutes(seg["time"])
                local_segs[(u, v)] = int(t)

    if isinstance(segs, dict):
        for k, arr in segs.items():
            ingest(norm_key(k), arr)
    elif isinstance(segs, list):
        for bucket in segs:
            if isinstance(bucket, dict):
                for k, arr in bucket.items():
                    ingest(norm_key(k), arr)

    for (a, b), t in list(local_segs.items()):
        if (b, a) not in local_segs:
            local_segs[(b, a)] = t

    return topo, local_segs


def build_station_y(topo: List[str], local_segs: Dict[Tuple[str, str], int]) -> Dict[str, float]:
    if not topo:
        return {}
    y = {topo[0]: 0.0}
    acc = 0.0
    for i in range(len(topo) - 1):
        u, v = topo[i], topo[i + 1]
        dt = local_segs.get((u, v), 1)  # 未定義は等間隔=1分
        acc += float(dt)
        y[v] = acc
    return y


def derive_topology_and_y(bundle: Dict[str, np.ndarray], station_yaml_path: Optional[str]) -> Tuple[List[str], Dict[str, float]]:
    topo = None
    local_segs: Dict[Tuple[str, str], int] = {}

    if bundle.get("topology") is not None:
        topo = list(bundle["topology"])

    if (topo is None or not local_segs) and station_yaml_path and os.path.exists(station_yaml_path):
        topo2, local_segs2 = load_station_topology_and_local_times(station_yaml_path)
        if topo is None:
            topo = topo2
        local_segs = local_segs2

    if topo is None:
        st_set = set(bundle["depart_station"]).union(set(bundle["arrive_station"]))
        topo = sorted(st_set)

    ymap = build_station_y(topo, local_segs)
    return topo, ymap
