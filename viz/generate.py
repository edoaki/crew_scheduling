from __future__ import annotations
from typing import Tuple
from pathlib import Path

def generate_and_save(station_yaml: str, train_yaml: str, out_npz: str) -> Tuple[bool, str]:
    try:
        from timetable.simulator import generate_timetable
    except Exception as e:
        return False, f"timetable.simulator の import に失敗: {e}"
    try:
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        rows = generate_timetable(
            station_yaml_path=station_yaml,
            train_yaml_path=train_yaml,
            save_path=out_npz,
        )
        return True, f"{len(rows)} rows"
    except Exception as e:
        return False, f"生成中エラー: {e}"
