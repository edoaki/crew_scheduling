# tokens.py
# station.yaml / train.yaml に合わせたロード・Network構築・Token生成・パラメータ生成

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import yaml

from .core_types import (
    Network, Station, Token, Direction, Service,
    parse_time_hhmm, parse_mmss_like
)


def load_configs(station_yaml_path: str, train_yaml_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(station_yaml_path, "r", encoding="utf-8") as f:
        station_raw = yaml.safe_load(f)
    with open(train_yaml_path, "r", encoding="utf-8") as f:
        train_raw = yaml.safe_load(f)
    return station_raw, train_raw


def _sv(val: str) -> Service:
    return Service(val)


def _dv(val: str) -> Direction:
    return Direction(val)


def build_network(station_raw: Dict[str, Any]) -> Network:
    stations: Dict[str, Station] = {}
    for s in station_raw["stations"]:
        stations[s["id"]] = Station(
            id=s["id"],
            express_stop=bool(s.get("express_stop", False)),
            depot=bool(s.get("depot", False)),
            crew_rest=bool(s.get("crew_rest", False)),
            turnback=bool(s.get("turnback", False)),
        )

    local_segments: Dict[Tuple[str, str], int] = {}
    rapid_segments: Dict[Tuple[str, str], int] = {}

    def _norm_key(k: Any) -> str:
        # 例: "-local" / " local " / "LOCAL" などを "local" に正規化
        return str(k).strip().lstrip("-").strip().lower()

    def _ingest(key: str, arr: Any) -> None:
        if not isinstance(arr, list):
            return
        if key == "local":
            for seg in arr:
                u, v = seg["from"], seg["to"]
                t = parse_mmss_like(seg["time"])
                local_segments[(u, v)] = t
        elif key == "rapid":
            for seg in arr:
                u, v = seg["from"], seg["to"]
                t = parse_mmss_like(seg["time"])
                rapid_segments[(u, v)] = t

    segs = station_raw.get("segments", {})
    if isinstance(segs, dict):
        for k, arr in segs.items():
            _ingest(_norm_key(k), arr)
    elif isinstance(segs, list):
        for bucket in segs:
            if isinstance(bucket, dict):
                for k, arr in bucket.items():
                    _ingest(_norm_key(k), arr)

    # 片方向だけ書かれていても両方向参照できるよう対称補完
    for (a, b), t in list(local_segments.items()):
        if (b, a) not in local_segments:
            local_segments[(b, a)] = t
    for (a, b), t in list(rapid_segments.items()):
        if (b, a) not in rapid_segments:
            rapid_segments[(b, a)] = t

    topology = list(station_raw.get("topology", {}).get("ordered_stations", []))

    def _parse_toward(val: Any) -> Optional[str]:
        if not isinstance(val, str) or not topology:
            return None
        s = val.strip()
        if "→" in s:
            cand = s.split("→", 1)[1].strip()
        elif "->" in s:
            cand = s.split("->", 1)[1].strip()
        else:
            cand = s
        return cand if cand in topology else None

    meta_dirs = station_raw.get("metadata", {}).get("directions", {})
    first = topology[0] if topology else ""
    last = topology[-1] if topology else ""
    up_toward = _parse_toward(meta_dirs.get("up")) or last
    down_toward = _parse_toward(meta_dirs.get("down")) or first

    return Network(
        stations=stations,
        local_segments=local_segments,
        rapid_segments=rapid_segments,
        topology=topology,
        up_toward=up_toward,
        down_toward=down_toward,
    )


def make_tokens_from_config(train_raw: Dict[str, Any]) -> List[Token]:
    tokens: List[Token] = []
    for item in train_raw.get("dispatch", []):
        station = item["station"]
        for win in item.get("windows", []):
            start = parse_time_hhmm(win["start"])
            end = parse_time_hhmm(win["end"])
            count = int(win["count"])
            direction = _dv(win["direction"])
            service = _sv(win.get("service", "local"))
            if count <= 0 or end <= start:
                continue
            span = end - start
            step = span / count
            for i in range(count):
                target = int(round(start + (i + 0.5) * step))
                tokens.append(Token(
                    station=station,
                    direction=direction,
                    service=service,
                    release_time=target,
                    deadline=end,
                    target=target,
                ))
    tokens.sort(key=lambda t: (t.release_time, t.station, t.direction, t.service))
    return tokens


def build_params(station_raw: Dict[str, Any], train_raw: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    # min_times（station.yaml）
    mt = station_raw.get("min_times", {})
    dwell_default = parse_mmss_like(mt.get("dwell", "00:01"))
    turnback_default = parse_mmss_like(mt.get("turnback", "00:07"))
    mto = station_raw.get("min_times_overrides", {})
    dwell_over = {k: parse_mmss_like(v) for k, v in mto.get("dwell", {}).items()}
    turn_over = {k: parse_mmss_like(v) for k, v in mto.get("turnback", {}).items()} if "turnback" in mto else {}

    params["dwell_default"] = dwell_default
    params["turnback_default"] = turnback_default
    params["dwell_overrides"] = dwell_over
    params["turnback_overrides"] = turn_over

    # headway（service単位のグローバル、station.yaml）
    hw_by_service: Dict[str, Dict[str, int]] = {}
    for hw in station_raw.get("headway", []):
        sv = hw["service"]
        after_stop = parse_mmss_like(hw.get("after_stop", "00:00"))
        after_turn = parse_mmss_like(hw.get("after_turnback", str(after_stop)))
        after_disp = parse_mmss_like(hw.get("after_dispatch", str(after_stop)))
        hw_by_service[sv] = {
            "after_stop": after_stop,
            "after_turnback": after_turn,
            "after_dispatch": after_disp,
        }
    params["headway_by_service"] = hw_by_service
    params["headway_default"] = {"after_stop": 0, "after_turnback": 0, "after_dispatch": 0}

    # stabling（train.yaml）
    stab_rules = []
    for i, r in enumerate(train_raw.get("stabling", [])):
        stab_rules.append({
            "id": f"stab_{i}",
            "station": r["station"],
            "direction_inbound": r["direction_inbound"],
            "start": parse_time_hhmm(r["start"]),
            "target_count": int(r["target_count"]),
            "probability": float(r["probability"]),
            "min_interval": parse_mmss_like(r.get("min_interval", "00:00")),
            "consumed": 0,
            "last_time": None,
        })
    params["stabling_rules"] = stab_rules

    # short_turn（train.yaml）: service 指定あり
    st_rules = []
    for i, r in enumerate(train_raw.get("short_turn", [])):
        st_rules.append({
            "id": f"short_{i}",
            "station": r["station"],
            "direction": r["direction"],
            "service": r.get("service"),  # 省略可にしておく
            "start": parse_time_hhmm(r["start"]),
            "target_count": int(r["target_count"]),
            "probability": float(r["probability"]),
            "min_interval": parse_mmss_like(r.get("min_interval", "00:00")),
            "consumed": 0,
            "last_time": None,
        })
    params["short_turn_rules"] = st_rules

    # service_conversion（train.yaml）: リスト直下
    sc_rules = []
    for i, r in enumerate(train_raw.get("service_conversion", [])):
        sc_rules.append({
            "id": f"sc_{i}",
            "station": r["station"],
            "direction": r["direction"],
            "start": parse_time_hhmm(r["start"]),
            "from_type": r["from_type"],
            "to_type": r["to_type"],
            "target_count": int(r["target_count"]),
            "probability": float(r["probability"]),
            "min_interval": parse_mmss_like(r.get("min_interval", "00:00")),
            "consumed": 0,
            "last_time": None,
        })
    params["service_conversion_rules"] = sc_rules

    return params
