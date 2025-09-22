import yaml
import numpy as np

def load_yaml(path: str) -> dict:
    """
    constraints.yaml を読み込み dict を返す。
    期待キー:
      - board_min
      - post_hitch_ready_min
      - max_hops
      - window_min
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"path : {path} yaml は辞書形式を想定しています")
    return raw

def add_crew_loader(crew_yaml_path: str):
    data = load_yaml(crew_yaml_path)
    add_crew = data["add_crew"]

    def _to_min(x):
        # 既に分(整数)ならそのまま
        if isinstance(x, (int, np.integer)):
            return int(x)
        # "HH:MM" 形式を想定（誤って "08;00" などでも ":" に正規化）
        s = str(x).strip().replace("；", ":").replace(";", ":")
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm)
    
    add_start = add_crew["additional_start"]
    add_end = add_crew["additional_end"]
    add_signoff = add_crew["additional_signoff_limit"]

    add_start_min = _to_min(add_start)
    add_end_min = _to_min(add_end)
    add_signoff_min = _to_min(add_signoff)

    return{
        "additional_start_min": add_start_min,
        "additional_end_min": add_end_min,
        "additional_signoff_limit_min": add_signoff_min,
    }


