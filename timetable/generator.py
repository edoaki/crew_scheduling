from __future__ import annotations
from typing import Tuple, Dict, Any, List
from pathlib import Path
import numpy as np
from typing import Optional
from timetable.simulator import generate_timetable  # fallback
from timetable.task_station_cache import load_constraints_yaml
from timetable.round_cache import build_round_cache

def _to_minutes(x):
    # int or "HH:MM"
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        x = x.strip()
        if ":" in x:
            hh, mm = x.split(":")
            return int(hh) * 60 + int(mm)
        # 万一 "530" のような文字数値が来た場合
        if x.isdigit():
            return int(x)
    raise TypeError(f"Unsupported time type: {type(x)} / value={x}")

def _rows_to_tt_arrays(rows: List[dict]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    def get(r, k):
        if isinstance(r, dict):
            return r.get(k)
        return getattr(r, k)

    N = len(rows)

    # まずはPythonリストに集めてから、最後に np.str_ 固定長Unicode配列へ変換
    train_ids_list: List[str] = []
    service_list: List[str] = []
    direction_list: List[str] = []

    dep_raw: List[str] = []
    arr_raw: List[str] = []

    dep_time = np.empty(N, dtype=np.int32)
    arr_time = np.empty(N, dtype=np.int32)

    for i, r in enumerate(rows):
        train_ids_list.append(str(get(r, "train_id")))
        service_list.append(str(get(r, "service")))
        direction_list.append(str(get(r, "direction")))

        ds = get(r, "depart_station")
        as_ = get(r, "arrive_station")
        dep_raw.append(str(ds))
        arr_raw.append(str(as_))

        dep_time[i] = _to_minutes(str(get(r, "depart_time")))
        arr_time[i] = _to_minutes(str(get(r, "arrive_time")))

    # 駅ラベル語彙（アルファベット順など固定順）
    unique_labels = sorted(set(dep_raw) | set(arr_raw))  # 例: ['A','B','C',...]
    station_label_vocab: List[str] = list(unique_labels)
    index: Dict[str, int] = {lab: i for i, lab in enumerate(station_label_vocab)}

    # エンコード
    depart_station = np.asarray([index[s] for s in dep_raw], dtype=np.int32)
    arrive_station = np.asarray([index[s] for s in arr_raw], dtype=np.int32)


    # 文字列列は固定長Unicodeで保存（allow_pickle不要にする）
    train_ids = np.array(train_ids_list, dtype=np.str_)
    service = np.array(service_list, dtype=np.str_)
    direction = np.array(direction_list, dtype=np.str_)

    tt = dict(
        train_ids=train_ids,
        depart_station=depart_station,
        arrive_station=arrive_station,
        depart_time=dep_time,
        arrive_time=arr_time,
        service=service,
        direction=direction,
    )
    return tt, station_label_vocab


def _build_round_from_round_id(round_id: np.ndarray, depart_time: np.ndarray) -> Dict[str, np.ndarray]:
    """
    round_id: shape [N] 1-based. Groups are 1..R.
    depart_time: shape [N] minutes.

    Returns:
      {"round_ptr":[R+1], "round_tt_idx":[sum], "round_anchor_min":[R]}
    """
    if round_id.size == 0:
        return dict(round_ptr=np.asarray([0], dtype=np.int32),
                    round_tt_idx=np.asarray([], dtype=np.int32),
                    round_anchor_min=np.asarray([], dtype=np.int32))
    rid = np.asarray(round_id, dtype=np.int64)
    N = rid.shape[0]
    idx = np.arange(N, dtype=np.int64)
    order = np.lexsort((idx, depart_time, rid))
    rid_sorted = rid[order]; tt_idx_sorted = idx[order]

    ptr = [0]; anchor = []; flat = []
    cur = rid_sorted[0]; start = 0
    for i in range(N):
        if rid_sorted[i] != cur:
            group_idx = tt_idx_sorted[start:i]
            flat.extend(group_idx.tolist())
            anchor.append(int(depart_time[group_idx].min()))
            ptr.append(len(flat))
            cur = rid_sorted[i]; start = i
    group_idx = tt_idx_sorted[start:N]
    flat.extend(group_idx.tolist())
    anchor.append(int(depart_time[group_idx].min()))
    ptr.append(len(flat))

    return dict(
        round_ptr=np.asarray(ptr, dtype=np.int32),
        round_tt_idx=np.asarray(flat, dtype=np.int32),
        round_anchor_min=np.asarray(anchor, dtype=np.int32),
    )

def generate_and_save(
    station_yaml: str,
    train_yaml: str,
    constraints_yaml : str,
    out_npz: str,
    *,
    seed: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    timetable生成の全体管理:
      1) simulator.generate_timetable で rows 生成（保存しない）
      2) time_table.round_cache.build_round_cache で round_id 等を算出
      3) time_table.task_station_cache.build_task_station_cache で task×station キャッシュを算出
      4) io_npz.save_timetable_bundle で単一NPZ保存

    board_min / post_hitch_ready_min / max_hops / window_min は
    constraints.yaml（--constraints）で与える。
    未指定キーは従来デフォルトにフォールバックする。
    """
    # 1) rows 生成
    

    try:
        rows = generate_timetable(
            station_yaml_path=station_yaml,
            train_yaml_path=train_yaml,
            seed=seed,
        )
        if rows is None:
            return False, "generate_timetable が None を返しました"
    except TypeError:
        rows = generate_timetable(station_yaml, train_yaml)
    except Exception as e:
        return False, f"生成中エラー: {e}"

    # constraints（YAML指定のみ／個別引数なし）
    DEFAULTS = {
        "board_min": 2,
        "post_hitch_ready_min": 5,
        "max_hops": 2,
        "window_min": 120,
    }
    conf = DEFAULTS.copy()
    try:
        
        raw = load_constraints_yaml(constraints_yaml)
        for k in DEFAULTS.keys():
            if k in raw and raw[k] is not None:
                conf[k] = int(raw[k])
    except Exception as e:
        return False, f"constraints 読込エラー: {e}"

    # rows -> tt/*（駅ラベルを語彙化）
    try:
        tt, station_label_vocab = _rows_to_tt_arrays(rows)
    except Exception as e:
        return False, f"rows 変換エラー: {e}"

    num_stations = len(station_label_vocab)

    # 2) round
    
    round_id, round_lock_task_ids, round_lock_ptr = build_round_cache(
        dep_station=tt["depart_station"],
        dep_time=tt["depart_time"],
        arr_station=tt["arrive_station"],
        arr_time=tt["arrive_time"],
        num_stations=num_stations,
    )
    round_arrays = _build_round_from_round_id(round_id, tt["depart_time"])

    # 3) cache（task×station）
    task_station_cache: Dict[str, np.ndarray] = {}
    try:
        from timetable.task_station_cache import build_task_station_cache
        (
            cache_task_ptr,
            cache_station_ids,
            cache_must_be_by_min,
            cache_is_hitch,
            cache_hops,
            cache_hitch_minutes,
            cache_path_ptr,
            cache_path_task_ids,
        ) = build_task_station_cache(
            dep_station=tt["depart_station"],
            dep_time=tt["depart_time"],
            arr_station=tt["arrive_station"],
            arr_time=tt["arrive_time"],
            board_min=int(conf["board_min"]),
            post_hitch_ready_min=int(conf["post_hitch_ready_min"]),
            num_stations=num_stations,
            max_hops=int(conf["max_hops"]),
            window_min=int(conf["window_min"]),
        )
        task_station_cache.update(dict(
            cache_task_ptr=cache_task_ptr,
            cache_station_ids=cache_station_ids,
            cache_must_be_by_min=cache_must_be_by_min,
            cache_is_hitch=cache_is_hitch,
            cache_hops=cache_hops,
            cache_hitch_minutes=cache_hitch_minutes,
            cache_path_ptr=cache_path_ptr,
            cache_path_task_ids=cache_path_task_ids,
        ))
    except Exception as e:
        return False, f"task_station 計算エラー: {e}"

    # 4) meta
    meta: Dict[str, Any] = dict(
        format="bundle_v1",
        num_stations=num_stations,
        board_min=int(conf["board_min"]),
        post_hitch_ready_min=int(conf["post_hitch_ready_min"]),
        max_hops=int(conf["max_hops"]),
        window_min=int(conf["window_min"]),
        seed=int(seed) if seed is not None else None,
        source_config=dict(station_yaml=str(station_yaml), train_yaml=str(train_yaml)),
        station_label_vocab=station_label_vocab,
    )
    if constraints_yaml:
        meta["source_constraints_yaml"] = str(constraints_yaml)

    # 5) 保存
    from utils.io_npz import save_timetable_bundle
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    id2lab = np.array(station_label_vocab, dtype=np.str_)
    tt_save = tt.copy()
    tt_save["depart_station"] = id2lab[tt["depart_station"]]
    tt_save["arrive_station"] = id2lab[tt["arrive_station"]]

    save_timetable_bundle(
        out_path=out_npz,
        tt_arrays=tt_save,
        meta=meta,
        task_station_cache=task_station_cache,
        round_arrays=round_arrays,
        topology=id2lab,
    )

    return True, f"保存完了: {out_npz} (rows={tt['depart_time'].shape[0]})"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--station", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--constraints", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    ok, msg = generate_and_save(
        a.station, a.train, a.constraints,a.out,
        seed=a.seed
    )
    print(("OK " if ok else "NG ") + msg)
