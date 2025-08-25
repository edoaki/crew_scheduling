# simulator.py
# 新形式（service別headway、rapidの飛び区間、up/down方位）対応の最小シミュレータ

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import heapq
import numpy as np
import random

from .core_types import (
    Network, Train, Token, TimetableRow,
    Direction, Service,
    parse_time_hhmm, format_time_hhmm,
    next_stop_station, travel_minutes, can_stop_here, opposite_direction
)
from .tokens import load_configs, build_network, make_tokens_from_config, build_params


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
    ctx.last_departure[last_key] = depart_time

    inbound_dir = train.direction
    stop_here = can_stop_here(ctx.network, v, train.service)
    dwell = _get_dwell(ctx, v) if stop_here else 0
    decision_time = arrive_time + dwell  # 判定のタイムスタンプ

    if _maybe_stabling(ctx, v, decision_time, inbound_dir):
        train.alive = False
        return

    turned = _maybe_short_turn(ctx, train, v, decision_time)
    _ = _maybe_service_conversion(ctx, train, v, decision_time)

    train.current_station = v

    if turned:
        tb = _get_turnback(ctx, v)
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


def save_timetable_npz(rows: List[TimetableRow], path: str) -> None:
    train_ids = np.array([r.train_id for r in rows])
    dep_st = np.array([r.depart_station for r in rows])
    arr_st = np.array([r.arrive_station for r in rows])
    dep_t = np.array([r.depart_time for r in rows], dtype=np.int32)
    arr_t = np.array([r.arrive_time for r in rows], dtype=np.int32)
    service = np.array([r.service.value for r in rows])
    direction = np.array([r.direction.value for r in rows])
    np.savez(path, train_ids=train_ids, depart_station=dep_st, arrive_station=arr_st,
             depart_time=dep_t, arrive_time=arr_t, service=service, direction=direction)


def generate_timetable(
    station_yaml_path: str,
    train_yaml_path: str,
    sim_start_hhmm: str = "04:00",
    sim_end_hhmm: str = "25:00",
    seed: Optional[int] = None,
    save_path: Optional[str] = None
) -> List[TimetableRow]:
    station_raw, train_raw = load_configs(station_yaml_path, train_yaml_path)
    network = build_network(station_raw)
    params = build_params(station_raw, train_raw)
    rng = random.Random(seed if seed is not None else 1234)

    tokens = make_tokens_from_config(train_raw)

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
    if save_path:
        save_timetable_npz(rows, save_path)
    return rows
