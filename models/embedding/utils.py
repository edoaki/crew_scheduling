import yaml
import torch


def _hhmm_to_minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def load_station_time_from_A(
    station_yaml_path: str,
    encoding_yaml_path: str,
) -> torch.Tensor:
    with open(station_yaml_path, "r", encoding="utf-8") as f:
        st = yaml.safe_load(f)

    segs_local = None
    if isinstance(st.get("segments"), dict):
        segs_local = st["segments"].get("local") or st["segments"].get("-local")
    if not segs_local:
        raise ValueError("segments.local または segments.-local が見つかりません。")

    ordered = st["topology"]["ordered_stations"]
    try:
        a_idx = ordered.index("A")
    except ValueError:
        raise ValueError("topology.ordered_stations に 'A' が見つかりません。")

    edge_min = {(seg["from"], seg["to"]): _hhmm_to_minutes(seg["time"]) for seg in segs_local}

    prefix = [0]
    for i in range(len(ordered) - 1):
        u, v = ordered[i], ordered[i + 1]
        if (u, v) not in edge_min:
            raise ValueError(f"local 区間所要が不足しています: {u} -> {v}")
        prefix.append(prefix[-1] + edge_min[(u, v)])

    t_from_A_map = {sid: abs(prefix[i] - prefix[a_idx]) for i, sid in enumerate(ordered)}

    with open(encoding_yaml_path, "r", encoding="utf-8") as f:
        enc = yaml.safe_load(f)
    sid_map = enc["station_id"]

    t_vec = torch.zeros(len(sid_map), dtype=torch.float32)
    for sid, idx in sid_map.items():
        if sid not in t_from_A_map:
            raise ValueError(f"station {sid} が topology.ordered_stations に存在しません。")
        t_vec[idx] = float(t_from_A_map[sid])

    return t_vec

