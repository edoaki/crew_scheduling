import numpy as np
from utils.yaml_loader import load_yaml

def generate_initial_crew(crew_yaml_path, encoding):
    """
    返り値:
      {
        "start_station_idx": np.int32[C],
        "assignable_start_min": np.int32[C],
        "assignable_end_min": np.int32[C],
        "end_station_ids_rag": np.object_[C],  # 各乗務員の終点駅ID配列（np.int32, 可変長）
        "crew_slot_label": np.int8[C],         # am=0, pm=1
        "crew_signoff_limit_min": np.int32[C], # 各クルーのサインオフ上限（分）
      }
    """
    def _to_min(x):
        # 既に分(整数)ならそのまま
        if isinstance(x, (int, np.integer)):
            return int(x)
        # "HH:MM" 形式を想定（誤って "08;00" などでも ":" に正規化）
        s = str(x).strip().replace("；", ":").replace(";", ":")
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm)

    data = load_yaml(crew_yaml_path)
    time_labels = data["time_labels"]
    placements = data["placements"]

    # am/pm のスロット定義の取り出し（signoff_limit は signoff_limits の誤記にも対応）
    def slot_info(slot_key):
        info = time_labels[slot_key]
        start = _to_min(info["start"])
        end = _to_min(info["end"])
        s_limit_str = info.get("signoff_limit", info.get("signoff_limits"))
        if s_limit_str is None:
            raise KeyError(f'time_labels["{slot_key}"] に signoff_limit がありません')
        signoff_limit = _to_min(s_limit_str)
        return start, end, signoff_limit
    
    am_start, am_end, am_signoff = slot_info("am")
    pm_start, pm_end, pm_signoff = slot_info("pm")

    # 駅エンコード
    station_enc = encoding["station_id"]
    if station_enc is None:
        raise KeyError("encoding に 'station_enc'（駅名->ID の辞書）が必要です")

    # 集約用バッファ
    start_station_idx = []
    assignable_start_min = []
    assignable_end_min = []
    end_station_ids_rag = []
    crew_slot_label = []          # am=0, pm=1
    crew_signoff_limit_min = []   # 各クルーのサインオフ上限(分)

    slot_to_id = {"am": 0, "pm": 1}

    for pl in placements:
        slot = str(pl["slot"]).lower()
        if slot not in slot_to_id:
            raise ValueError(f'未知のslot "{slot}" （am/pm だけを許可）')

        if slot == "am":
            t_start, t_end, s_limit = am_start, am_end, am_signoff
        else:
            t_start, t_end, s_limit = pm_start, pm_end, pm_signoff

        s_id = int(station_enc[pl["station"]])
        allowed = pl.get("allowed_signoff", [])
        end_ids = np.array([station_enc[a] for a in allowed], dtype=np.int32)

        for _ in range(int(pl["count"])):
            start_station_idx.append(s_id)
            assignable_start_min.append(t_start)
            assignable_end_min.append(t_end)
            end_station_ids_rag.append(end_ids.copy())
            crew_slot_label.append(slot_to_id[slot])
            crew_signoff_limit_min.append(s_limit)

    return {
        "start_station_idx": np.asarray(start_station_idx, dtype=np.int32),
        "assignable_start_min": np.asarray(assignable_start_min, dtype=np.int32),
        "assignable_end_min": np.asarray(assignable_end_min, dtype=np.int32),
        "end_station_ids_rag": np.asarray(end_station_ids_rag, dtype=object),
        "crew_slot_label": np.asarray(crew_slot_label, dtype=np.int8),
        "crew_signoff_limit_min": np.asarray(crew_signoff_limit_min, dtype=np.int32),
    }

