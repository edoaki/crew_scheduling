# timetable/io_npz.py
# ダイヤ保存・読み込みの専用モジュール
# - 単一NPZに「本体(tt/*)」「メタ(meta/*)」「将来のキャッシュ(feat/*)」を同梱
# - 列車ごとの索引は保存しない（可視化で都度計算）

from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
import json
import os

from .core_types import TimetableRow


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_timetable_bundle(
    path: str,
    rows: List[TimetableRow],
    topology: List[str],
    meta_json: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    rowsをNPZへ保存。将来のキャッシュ(features)も同梱可能。
    - 本体: tt/*
    - メタ:  meta/topology, meta/station_index, meta/json
    - 追加:  feat/*（任意、常にセット利用が前提）
    """
    _ensure_parent_dir(path)

    train_ids = np.array([r.train_id for r in rows])
    dep_st = np.array([r.depart_station for r in rows])
    arr_st = np.array([r.arrive_station for r in rows])
    dep_t = np.array([r.depart_time for r in rows], dtype=np.int32)
    arr_t = np.array([r.arrive_time for r in rows], dtype=np.int32)
    service = np.array([r.service.value for r in rows])
    direction = np.array([r.direction.value for r in rows])

    topo = np.array(list(topology))
    st_index = {st: i for i, st in enumerate(topology)}
    # station_indexは可視化用に保存（駅→Y座標）
    station_index = np.array([st_index[s] for s in topology], dtype=np.int32)

    to_save: Dict[str, Any] = {
        "tt/train_ids": train_ids,
        "tt/depart_station": dep_st,
        "tt/arrive_station": arr_st,
        "tt/depart_time": dep_t,
        "tt/arrive_time": arr_t,
        "tt/service": service,
        "tt/direction": direction,
        "meta/topology": topo,
        "meta/station_index": station_index,
        "meta/json": np.frombuffer(
            json.dumps(meta_json or {}, ensure_ascii=False).encode("utf-8"), dtype=np.uint8
        ),
        # 将来: feat/* をフラットに追加
    }

    if features:
        for k, v in features.items():
            to_save[f"feat/{k}"] = v

    np.savez(path, **to_save)


def load_timetable_bundle(path: str, mmap: bool = True) -> Dict[str, Any]:
    """
    NPZを読み込み、辞書で返す。
    旧形式（キーがプレフィクス無し）の後方互換にも対応。
    """
    loader = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    keys = list(loader.keys())

    def pick(name: str, fallback: Optional[str] = None):
        if name in loader:
            return loader[name]
        if fallback and fallback in loader:
            return loader[fallback]
        return None

    data = {
        "train_ids": pick("tt/train_ids", "train_ids"),
        "depart_station": pick("tt/depart_station", "depart_station"),
        "arrive_station": pick("tt/arrive_station", "arrive_station"),
        "depart_time": pick("tt/depart_time", "depart_time"),
        "arrive_time": pick("tt/arrive_time", "arrive_time"),
        "service": pick("tt/service", "service"),
        "direction": pick("tt/direction", "direction"),
        "topology": pick("meta/topology"),
        "station_index": pick("meta/station_index"),
        "meta_json": pick("meta/json"),
        # feat/* は動的に抽出
    }

    # メタJSONの復元
    if data["meta_json"] is not None:
        try:
            b = bytes(data["meta_json"])
            data["meta"] = json.loads(b.decode("utf-8"))
        except Exception:
            data["meta"] = {}
    else:
        data["meta"] = {}

    # features
    features: Dict[str, Any] = {}
    for k in keys:
        if k.startswith("feat/"):
            features[k[5:]] = loader[k]
    data["features"] = features

    return data
