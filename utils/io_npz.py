# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional
import os
import json
import numpy as np

def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _to_unicode_array(a: np.ndarray) -> np.ndarray:
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    if a.dtype == object:
        try:
            return np.array(a.tolist(), dtype=np.str_)
        except Exception:
            return np.array([str(x) for x in a.tolist()], dtype=np.str_)
    return a

def save_timetable_bundle(
    out_path: str,
    tt_arrays: Dict[str, np.ndarray],
    meta: Optional[Dict[str, Any]] = None,
    task_station_cache: Optional[Dict[str, np.ndarray]] = None,
    round_arrays: Optional[Dict[str, np.ndarray]] = None,
    topology: Optional[np.ndarray] = None,  # ←追加
) -> None:
    """
    単一NPZ（圧縮）に保存する。
      - 時刻表カラムは 'tt/<name>'
      - メタは 'meta/json'（JSONのUTF-8バイト列）
      
      - ラウンド関連は 'round/<name>'
      - キャッシュ関連は 'task_station_cache/<name>'
    """
    _ensure_parent_dir(out_path)
    save_dict: Dict[str, Any] = {}

    # timetable arrays（object配列はUnicodeへ正規化）
    for k, v in (tt_arrays or {}).items():
        a = np.asarray(v)
        # 文字列列が object のままだと allow_pickle=False で読めないので固定長Unicodeに変換
        if a.dtype == object:
            try:
                a = a.astype(np.str_)
            except Exception:
                # 数値混在などで変換不可なら元のまま（通常 tt/* は数値か文字列のはず）
                pass
        save_dict[f"tt/{k}"] = a


    # meta
    if meta is not None:
        try:
            meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        except Exception:
            meta_bytes = b"{}"
        save_dict["meta/json"] = np.frombuffer(meta_bytes, dtype=np.uint8)
    
    # features / round は数値想定だが一応object回避
    if task_station_cache:
        for k, v in task_station_cache.items():
            arr = np.asarray(v)
            save_dict[f"task_station_cache/{k}"] = _to_unicode_array(arr)
        
    if round_arrays:
        for k, v in round_arrays.items():
            arr = np.asarray(v)
            save_dict[f"round/{k}"] = _to_unicode_array(arr)
            
    if topology is not None:
        save_dict["topology"] = np.asarray(topology, dtype=np.str_)

    np.savez_compressed(out_path, **save_dict)

def _strip_namespace_prefix(d, namespace):
    pre = f"{namespace}/"
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and k.startswith(pre):
            out[k[len(pre):]] = v
        else:
            out[k] = v
    return out

def _extract_npz(z, sanitize_object: bool) -> Dict[str, Any]:
    keys = list(z.keys())

    # tt
    data_tt = {}
    for k in keys:
        if k.startswith("tt/"):
            arr = z[k]
            if sanitize_object and getattr(arr, "dtype", None) == object:
                arr = _to_unicode_array(arr)
            data_tt[k] = arr

    # meta
    meta = {}
    if "meta/json" in keys:
        try:
            meta = json.loads(bytes(z["meta/json"]).decode("utf-8"))
        except Exception:
            meta = {}

    # features
    task_station_cache = {}
    for k in keys:
        if k.startswith("task_station_cache/"):
            arr = z[k]
            if sanitize_object and getattr(arr, "dtype", None) == object:
                arr = _to_unicode_array(arr)
            task_station_cache[k] = arr

    # round
    rounds = {}
    for k in keys:
        if k.startswith("round/"):
            arr = z[k]
            if sanitize_object and getattr(arr, "dtype", None) == object:
                arr = _to_unicode_array(arr)
            rounds[k] = arr

    return data_tt, meta, task_station_cache,  rounds


def load_timetable_bundle(path: str, strip_namespace: bool = True):
    """
    まず allow_pickle=False で読み、object配列エラーの場合のみ
    allow_pickle=True にフォールバックして文字配列はUnicodeへ変換する。
    読み出した後、strip_namespace=True なら 'tt/' や 'round/' 等の
    名前空間プレフィックスを取り除く。
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            tt, meta, task_station_cache, rounds = _extract_npz(z, sanitize_object=False)
    except ValueError as e:
        msg = str(e)
        if "Object arrays cannot be loaded" not in msg and "allow_pickle=False" not in msg:
            raise
        with np.load(path, allow_pickle=True) as z:
            tt, meta, task_station_cache, rounds = _extract_npz(z, sanitize_object=True)

    if strip_namespace:
        tt = _strip_namespace_prefix(tt, "tt")
        rounds = _strip_namespace_prefix(rounds, "round")
        task_station_cache = _strip_namespace_prefix(task_station_cache, "task_station_cache")

    return tt, meta, task_station_cache, rounds

from typing import Dict, Any, List
import yaml
def station_order_from_config(station_yaml_path: str) -> List[str]:
    with open(station_yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [s["id"] for s in raw["stations"]]