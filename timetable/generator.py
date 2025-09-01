from __future__ import annotations
from typing import Tuple, Dict, Any, List
from pathlib import Path
import numpy as np
from typing import Optional
from timetable.simulator import generate_timetable  # fallback
from timetable.task_station_cache import load_constraints_yaml
from timetable.round_cache import build_round_cache


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
    

    
    tt,station_label_vocab = generate_timetable(
        station_yaml_path=station_yaml,
        train_yaml_path=train_yaml,
        seed=seed,
    )
        
    num_stations = len(station_label_vocab)

    # 2) round
    round_arrays = build_round_cache(
        dep_station=tt["depart_station"],
        dep_time=tt["depart_time"],
        arr_station=tt["arrive_station"],
        arr_time=tt["arrive_time"],
        num_stations=num_stations,
    )

    contraints = load_constraints_yaml(constraints_yaml)
    
    # 3) cache（task×station）
    from timetable.task_station_cache import build_task_station_cache
    task_station_cache= build_task_station_cache(
        dep_station=tt["depart_station"],
        dep_time=tt["depart_time"],
        arr_station=tt["arrive_station"],
        arr_time=tt["arrive_time"],
        board_min=int(contraints["board_min"]),
        post_hitch_ready_min=int(contraints["post_hitch_ready_min"]),
        num_stations=num_stations,
        max_hops=int(contraints["max_hops"]),
        window_min=int(contraints["window_min"]),
    )

    # 4) meta
    meta: Dict[str, Any] = dict(
        format="bundle_v1",
        num_stations=num_stations,
        board_min=int(contraints["board_min"]),
        post_hitch_ready_min=int(contraints["post_hitch_ready_min"]),
        max_hops=int(contraints["max_hops"]),
        window_min=int(contraints["window_min"]),
        seed=int(seed) if seed is not None else None,
        source_config=dict(station_yaml=str(station_yaml), train_yaml=str(train_yaml)),
        station_label_vocab=station_label_vocab,
    )

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
