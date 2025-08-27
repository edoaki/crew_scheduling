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
    features: Optional[Dict[str, np.ndarray]] = None,
    round_arrays: Optional[Dict[str, np.ndarray]] = None,
    topology: Optional[np.ndarray] = None,  # ←追加
) -> None:
    """
    単一NPZ（圧縮）に保存する。
      - 時刻表カラムは 'tt/<name>'
      - メタは 'meta/json'（JSONのUTF-8バイト列）
      - 追加特徴は 'feat/<name>'
      - ラウンド関連は 'round/<name>'
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
    if features:
        for k, v in features.items():
            arr = np.asarray(v)
            save_dict[f"feat/{k}"] = _to_unicode_array(arr)
    if round_arrays:
        for k, v in round_arrays.items():
            arr = np.asarray(v)
            save_dict[f"round/{k}"] = _to_unicode_array(arr)
            
    if topology is not None:
        save_dict["topology"] = np.asarray(topology, dtype=np.str_)


    np.savez_compressed(out_path, **save_dict)

def _extract_npz(z, sanitize_object: bool) -> Dict[str, Any]:
    keys = list(z.keys())

    # tt
    data_tt = {}
    for k in keys:
        if k.startswith("tt/"):
            arr = z[k]
            if sanitize_object and getattr(arr, "dtype", None) == object:
                arr = _to_unicode_array(arr)
            data_tt[k[3:]] = arr

    # meta
    meta = {}
    if "meta/json" in keys:
        try:
            meta = json.loads(bytes(z["meta/json"]).decode("utf-8"))
        except Exception:
            meta = {}

    # features
    feats = {}
    for k in keys:
        if k.startswith("feat/"):
            arr = z[k]
            if sanitize_object and getattr(arr, "dtype", None) == object:
                arr = _to_unicode_array(arr)
            feats[k[5:]] = arr

    # round
    rounds = {}
    for k in keys:
        if k.startswith("round/"):
            arr = z[k]
            if sanitize_object and getattr(arr, "dtype", None) == object:
                arr = _to_unicode_array(arr)
            rounds[k[6:]] = arr

    return {"tt": data_tt, "meta": meta, "features": feats, "round": rounds}

def load_timetable_bundle(path: str) -> Dict[str, Any]:
    """
    まず allow_pickle=False で読み、object配列エラーの場合のみ
    allow_pickle=True にフォールバックして文字配列はUnicodeへ変換する。
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            return _extract_npz(z, sanitize_object=False)
    except ValueError as e:
        msg = str(e)
        if "Object arrays cannot be loaded" not in msg and "allow_pickle=False" not in msg:
            raise
    # フォールバック（古いNPZ互換）
    with np.load(path, allow_pickle=True) as z:
        return _extract_npz(z, sanitize_object=True)
