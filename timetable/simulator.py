# simulator.py
# 新形式（service別headway、rapidの飛び区間、up/down方位）対応の最小シミュレータ

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import heapq
import numpy as np
import random
import torch
from tensordict import TensorDict
import yaml


from .core_types import (
    Network, Train, Token, TimetableRow,
    Direction, Service,
    parse_time_hhmm, format_time_hhmm,
    next_stop_station, travel_minutes, can_stop_here, opposite_direction
)
from .tokens import build_network, make_tokens_from_config, build_params
from utils.yaml_loader import load_yaml


@dataclass
class TimetableRecorder:
    rows: List[TimetableRow] = field(default_factory=list)

    def record(self, row: TimetableRow) -> None:
        self.rows.append(row)

    def to_sorted(self) -> List[TimetableRow]:
        return sorted(self.rows, key=lambda r: r.depart_time)


@dataclass
class SimContext:
    clock: int
    network: Network
    params: Dict[str, Any]
    rng: random.Random
    trains: Dict[str, Train] = field(default_factory=dict)
    ready: List[Tuple[int, str]] = field(default_factory=list)
    recorder: TimetableRecorder = field(default_factory=TimetableRecorder)
    # 駅×種別×方向ごとの最後の出発時刻
    last_departure: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    row_extras: Dict[Tuple[str, int, int], Dict[str, int]] = field(default_factory=dict)

def _get_headway(ctx: SimContext, service: Service, mode: str) -> int:
    hmap = ctx.params["headway_by_service"]
    base = ctx.params["headway_default"]
    h = hmap.get(service.value, base)
    return int(h.get(mode, base.get(mode, 0)))


def _get_dwell(ctx: SimContext, station: str) -> int:
    return int(ctx.params["dwell_overrides"].get(station, ctx.params["dwell_default"]))


def _get_turnback(ctx: SimContext, station: str) -> int:
    return int(ctx.params["turnback_overrides"].get(station, ctx.params["turnback_default"]))


def materialize_initial_trains_from_tokens(ctx: SimContext, tokens: List[Token], sim_start: int, sim_end: int) -> None:
    seq = 0
    for t in tokens:
        if t.release_time >= sim_end:
            continue
        release = max(t.release_time, sim_start)
        next_st = next_stop_station(ctx.network, t.station, t.direction, t.service)
        if next_st is None:
            continue
        train_id = f"T{seq:06d}"
        seq += 1
        train = Train(
            id=train_id,
            service=t.service,
            direction=t.direction,
            current_station=t.station,
            next_station=next_st,
            next_action_time=release,
            meta={"depart_mode": "after_dispatch", "deadline": t.deadline},
        )
        ctx.trains[train_id] = train
        heapq.heappush(ctx.ready, (train.next_action_time, train.id))


def _maybe_stabling(ctx: SimContext, station: str, timestamp: int, inbound_dir: Direction) -> bool:
    for r in ctx.params["stabling_rules"]:
        if r["station"] != station:
            continue
        if r["direction_inbound"] != inbound_dir.value:
            continue
        if timestamp < r["start"]:
            continue
        if r["consumed"] >= r["target_count"]:
            continue
        lt = r["last_time"]
        if lt is not None and timestamp - lt < r["min_interval"]:
            continue
        if ctx.rng.random() <= r["probability"]:
            r["consumed"] += 1
            r["last_time"] = timestamp
            return True
    return False


def _maybe_short_turn(ctx: SimContext, train: Train, station: str, timestamp: int) -> bool:
    for r in ctx.params["short_turn_rules"]:
        if r["station"] != station:
            continue
        if r["direction"] != train.direction.value:
            continue
        sv_req = r.get("service")
        if sv_req is not None and sv_req != train.service.value:
            continue
        if timestamp < r["start"]:
            continue
        if r["consumed"] >= r["target_count"]:
            continue
        lt = r["last_time"]
        if lt is not None and timestamp - lt < r["min_interval"]:
            continue
        if ctx.rng.random() <= r["probability"]:
            r["consumed"] += 1
            r["last_time"] = timestamp
            train.direction = opposite_direction(train.direction)
            return True
    return False


def _maybe_service_conversion(ctx: SimContext, train: Train, station: str, timestamp: int) -> bool:
    for r in ctx.params["service_conversion_rules"]:
        if r["station"] != station:
            continue
        if r["direction"] != train.direction.value:
            continue
        if timestamp < r["start"]:
            continue
        if r["consumed"] >= r["target_count"]:
            continue
        if r["from_type"] != train.service.value:
            continue
        lt = r["last_time"]
        if lt is not None and timestamp - lt < r["min_interval"]:
            continue
        if ctx.rng.random() <= r["probability"]:
            r["consumed"] += 1
            r["last_time"] = timestamp
            train.service = Service(r["to_type"])
            return True
    return False


def step_one_train(ctx: SimContext, train_id: str) -> None:
    train = ctx.trains.get(train_id)
    if not train or not train.alive:
        return

    u = train.current_station
    v = train.next_station
    if u is None or v is None:
        train.alive = False
        return

    mode = train.meta.get("depart_mode", "after_stop")
    headway = _get_headway(ctx, train.service, mode)
    last_key = (u, train.service.value, train.direction.value)
    last_dep = ctx.last_departure.get(last_key, -10**9)

    candidate = train.next_action_time
    depart_time = max(candidate, last_dep + headway)

    if mode == "after_dispatch":
        deadline = int(train.meta.get("deadline", depart_time))
        if depart_time > deadline:
            train.alive = False
            return

    travel = travel_minutes(ctx.network, u, v, train.service)
    arrive_time = depart_time + travel

    ctx.recorder.record(TimetableRow(
        train_id=train.id,
        depart_station=u,
        arrive_station=v,
        depart_time=depart_time,
        arrive_time=arrive_time,
        service=train.service,
        direction=train.direction,
    ))
    key = (train.id, depart_time, arrive_time)
    ctx.row_extras[key] = {
        "is_dispatch_task": int(mode == "after_dispatch"),
        "is_depart_from_turnback": int(mode == "after_turnback"),
        "is_arrival_before_turnback": 0,
        "is_stabling_at_arrival": 0,
        "event_complete_time": -1,  # 後段で折返/収納が決まったら上書き
    }

    ctx.last_departure[last_key] = depart_time

    inbound_dir = train.direction
    stop_here = can_stop_here(ctx.network, v, train.service)
    dwell = _get_dwell(ctx, v) if stop_here else 0
    decision_time = arrive_time + dwell  # 判定のタイムスタンプ

    if _maybe_stabling(ctx, v, decision_time, inbound_dir):
        key = (train.id, depart_time, arrive_time)
        e = ctx.row_extras.get(key)
        if e is not None:
            e["is_stabling_at_arrival"] = 1
            e["event_complete_time"] = decision_time  # = arrive_time + dwell
        train.alive = False
        return


    turned = _maybe_short_turn(ctx, train, v, decision_time)
    _ = _maybe_service_conversion(ctx, train, v, decision_time)

    train.current_station = v

    if turned:
        tb = _get_turnback(ctx, v)
        key = (train.id, depart_time, arrive_time)
        e = ctx.row_extras.get(key)
        if e is not None:
            e["is_arrival_before_turnback"] = 1
            e["event_complete_time"] = arrive_time + dwell + tb  # 折返し「完了」時刻

        train.meta["depart_mode"] = "after_turnback"
        train.next_action_time = arrive_time + dwell + tb
        nxt = next_stop_station(ctx.network, v, train.direction, train.service)
        if nxt is None:
            train.alive = False
            return
        train.next_station = nxt
        if train.alive:
            heapq.heappush(ctx.ready, (train.next_action_time, train.id))
        return

    nxt = next_stop_station(ctx.network, v, train.direction, train.service)
    if nxt is None:
        # 自動折返し（turnback可なら）
        if ctx.network.stations[v].turnback:
            tb = _get_turnback(ctx, v)
            key = (train.id, depart_time, arrive_time)
            e = ctx.row_extras.get(key)
            if e is not None:
                e["is_arrival_before_turnback"] = 1
                e["event_complete_time"] = arrive_time + dwell + tb

            train.direction = opposite_direction(train.direction)
            train.meta["depart_mode"] = "after_turnback"
            train.next_action_time = arrive_time + dwell + tb
            nxt2 = next_stop_station(ctx.network, v, train.direction, train.service)
            if nxt2 is None:
                train.alive = False
                return
            train.next_station = nxt2
            if train.alive:
                heapq.heappush(ctx.ready, (train.next_action_time, train.id))
            return
        train.alive = False
        return

    train.next_station = nxt
    train.meta["depart_mode"] = "after_stop"  # 通過でも after_stop で統一
    train.next_action_time = arrive_time + dwell
    if train.alive:
        heapq.heappush(ctx.ready, (train.next_action_time, train.id))


def rows_to_td(rows, encoding: dict) -> TensorDict:
    # encoding: {"service": {...}, "direction": {...}, "station_id": {...}, "train_id_rule": {...}}
    service_tbl = encoding["service"]
    direction_tbl = encoding["direction"]
    station_tbl = encoding["station_id"]
    rule = encoding.get("train_id_rule", {})
    prefix = str(rule.get("prefix", "T"))

    def _get(r, k):
        if isinstance(r, dict):
            return r.get(k)
        return getattr(r, k)

    def _enum_name(x):
        # Enum -> .value（"local"/"up"...）、それ以外は小文字化した文字列
        try:
            v = x.value  # Enum
        except AttributeError:
            v = str(x)
        return v.strip().lower()

    def _enc_cat(name: str, x: str, tbl: dict) -> int:
        if x not in tbl:
            known = ", ".join(sorted(tbl))
            raise ValueError(f"{name} の未知カテゴリ: {x}. 既知: [{known}]")
        return int(tbl[x])

    def _enc_station(s: str) -> int:
        s = str(s).strip()
        if s not in station_tbl:
            known = ", ".join(sorted(station_tbl))
            raise ValueError(f"駅ID未知: {s}. 既知: [{known}]")
        return int(station_tbl[s])

    def _enc_train_id(tid) -> int:
        if isinstance(tid, (int, np.integer)):
            return int(tid)
        tid = str(tid)
        if not tid.startswith(prefix):
            raise ValueError(f"train_id prefix不一致: {tid} (expected prefix='{prefix}')")
        return int(tid[len(prefix):])

    N = len(rows)
    train_id = np.empty(N, dtype=np.int32)
    service = np.empty(N, dtype=np.int32)
    direction = np.empty(N, dtype=np.int32)
    depart_station = np.empty(N, dtype=np.int32)
    arrive_station = np.empty(N, dtype=np.int32)
    depart_time = np.empty(N, dtype=np.int32)
    arrive_time = np.empty(N, dtype=np.int32)

    for i, r in enumerate(rows):
        train_id[i] = _enc_train_id(_get(r, "train_id"))
        service[i] = _enc_cat("service", _enum_name(_get(r, "service")), service_tbl)
        direction[i] = _enc_cat("direction", _enum_name(_get(r, "direction")), direction_tbl)
        depart_station[i] = _enc_station(_get(r, "depart_station"))
        arrive_station[i] = _enc_station(_get(r, "arrive_station"))
        depart_time[i] = int(_get(r, "depart_time"))
        arrive_time[i] = int(_get(r, "arrive_time"))

    data = {
        "train_id": np.asarray(train_id, dtype=np.int32),
        "service": np.asarray(service, dtype=np.int32),
        "direction": np.asarray(direction, dtype=np.int32),
        "depart_station": np.asarray(depart_station, dtype=np.int32),
        "arrive_station": np.asarray(arrive_station, dtype=np.int32),
        "depart_time": np.asarray(depart_time, dtype=np.int32),
        "arrive_time": np.asarray(arrive_time, dtype=np.int32),
    }
    return data
def rows_to_td_with_events(
    rows,
    encoding: dict,
    extras: Optional[Dict[Tuple[str, int, int], Dict[str, int]]] = None,
):
    base = rows_to_td(rows, encoding)  # 既存の変換をそのまま利用
    N = len(rows)

    is_dispatch_task = np.zeros(N, dtype=np.int32)
    is_depart_from_turnback = np.zeros(N, dtype=np.int32)
    is_arrival_before_turnback = np.zeros(N, dtype=np.int32)
    is_stabling_at_arrival = np.zeros(N, dtype=np.int32)
    event_complete_time = np.full(N, -1, dtype=np.int32)  # 中間計算用
    next_event_time_from_depart = np.full(N, -1, dtype=np.int32)

    # 行インデックスをtrain_idごとに束ねる（次イベント探索用）
    per_train_indices: Dict[str, List[int]] = {}

    def _get(r, k):
        if isinstance(r, dict):
            return r.get(k)
        return getattr(r, k)

    # まず各行のフラグをextrasから埋め、trainごとの並びを作る
    for i, r in enumerate(rows):
        tid = str(_get(r, "train_id"))
        dt = int(_get(r, "depart_time"))
        at = int(_get(r, "arrive_time"))
        per_train_indices.setdefault(tid, []).append(i)

        if extras is not None:
            e = extras.get((tid, dt, at))
        else:
            e = None

        if e is not None:
            is_dispatch_task[i] = int(e.get("is_dispatch_task", 0))
            is_depart_from_turnback[i] = int(e.get("is_depart_from_turnback", 0))
            is_arrival_before_turnback[i] = int(e.get("is_arrival_before_turnback", 0))
            is_stabling_at_arrival[i] = int(e.get("is_stabling_at_arrival", 0))
            event_complete_time[i] = int(e.get("event_complete_time", -1))
        # e が無い場合はデフォルト(0/-1)のまま

    # 各train内で「未来の最初のevent_complete」を後ろ向きで引き当て
    depart_time = np.asarray(base["depart_time"], dtype=np.int32)
    for tid, idxs in per_train_indices.items():
        nearest = -1
        for j in reversed(idxs):
            if event_complete_time[j] != -1:
                nearest = j
            if nearest == -1:
                next_event_time_from_depart[j] = -1
            else:
                val = int(event_complete_time[nearest]) - int(depart_time[j])
                next_event_time_from_depart[j] = val if val >= 0 else 0

    # 追加キーを載せて返す（元のキーはそのまま）
    base["is_dispatch_task"] = is_dispatch_task
    base["is_depart_from_turnback"] = is_depart_from_turnback
    base["is_arrival_before_turnback"] = is_arrival_before_turnback
    base["is_stabling_at_arrival"] = is_stabling_at_arrival
    base["next_event_time_from_depart"] = next_event_time_from_depart
    return base


def generate_timetable(
    station_yaml_path: str,
    train_yaml_path: str,
    encoding: dict,
    seed: Optional[int] = None,
):
    station_raw = load_yaml(station_yaml_path)
    train_raw = load_yaml(train_yaml_path)
    network = build_network(station_raw)
    params = build_params(station_raw, train_raw)
    rng = random.Random(seed if seed is not None else 1234)

    tokens = make_tokens_from_config(train_raw)

    sim_start_hhmm = station_raw["sim_start"]
    sim_end_hhmm = station_raw["sim_end"]

    start = parse_time_hhmm(sim_start_hhmm)
    end = parse_time_hhmm(sim_end_hhmm)

    ctx = SimContext(clock=start, network=network, params=params, rng=rng)
    materialize_initial_trains_from_tokens(ctx, tokens, start, end)

    while ctx.ready and ctx.clock < end:
        t, train_id = heapq.heappop(ctx.ready)
        if t > end:
            break
        ctx.clock = max(ctx.clock, t)
        step_one_train(ctx, train_id)

    rows = ctx.recorder.to_sorted()
    tt = rows_to_td_with_events(rows, encoding, ctx.row_extras)
    train_num = int(np.unique(tt["train_id"]).size)
    return tt, train_num
    
