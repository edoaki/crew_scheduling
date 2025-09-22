import torch

from tensordict.tensordict import TensorDict

from typing import Any, Dict, List
from timetable.task_station_cache import build_task_station_cache
from timetable.simulator import generate_timetable  
from timetable.round_cache import build_round_cache
from timetable.crew import generate_initial_crew
from utils.yaml_loader import load_yaml
import numpy as np

class CrewARGenerator():
    def __init__(self,station_yaml,train_yaml,constraints_yaml,crew_yaml):
        self.station_yaml = station_yaml
        self.train_yaml = train_yaml
        self.constraints_yaml = constraints_yaml
        self.crew_yaml = crew_yaml
        self.encoding_yaml = "config/encoding.yaml"
       
    def generate(self, batch_size: int = 1) -> TensorDict:
        # 生データ生成
        tt, round_arrays, task_station_cache, crew, meta = self.generate_raw_data()

        # すべてマージ
        merged: Dict[str, Any] = {}
        merged.update(tt)
        merged.update(round_arrays)
        merged.update(task_station_cache)
        merged.update(crew)
        merged.update(meta)

        return merged

    # --- 生成される TensorDict の内容 ---
    # 行路の数　N、乗務員の人数 C、駅の数 S、ラウンド数 R、タスク数 T 

    # = バッチサイズ
    # tt (timetable・ダイヤ)  関連TDの意味
    # あるタスク は 1つの列車の運行(行路)を表し、電車・種別(local,rapid)・方向(up,down)・出発駅・到着駅・出発時間・到着時間の情報を持つ。
    # 'train_id' [N]: 行路ごとの列車ID
    # 'service' [N]: 行路ごとの列車種別ID
    # 'direction' [N]: 行路ごとの列車方向ID
    # 'depart_station' [N]: 行路ごとの出発駅ID
    # 'arrive_station' [N]: 行路ごとの到着駅ID
    # 'depart_time' [N]: 行路ごとの出発時刻(
    # 'arrive_time' [N]: 行路ごとの到着時刻(分)
    # 'is_dispatch_task' [N]: 行路ごとの出発駅が車庫か否か(bool)
    # 'is_depart_from_turnback' [N]: 行路ごとの出発
    # 'is_arrival_before_turnback' [N]: 行路ごとの到着駅が折り返し駅か否か(bool)
    # 'is_stabling_at_arrival' [N]: 行路ごとの到着駅が車庫か否か(bool)
    # 'next_event_time_from_depart' [N]: 行路ごとの出発から次のイベントまでの時間(分)
    
    # タイムステップはラウンドに対応し、各ラウンドは複数のタスクを含む。
    # 各ラウンドで、タスクには必ず1つの乗務員が割り当てられる。

    # round 関連TDの意味
    # round_first_task_id: [R]  各ラウンドの最初のタスクID
    # round_task_to_round: [T]  各タスクの所属ラウンドID
    # round_time: [R]   各ラウンドの開始時刻(分)
    
    # --- Cache キーの要約 ---
    # 各タスクに対応するためには 何時までに どの駅に居れば良いか？を示す情報を持つ。
    
    # station_ids [S]: 駅IDの配列（0始まり）
    # task_ids [T]: タスクIDの配列（0始まり）
    # must_be_by_min [T,S]: タスクtに対して、駅sに「この分までに居れば可」(int32)                 
    # is_hitch [T,S]: タスクtに対して、駅sに便乗を使うか(bool)
    # hops [T,S]: タスクtに対して、駅sに便乗本数(int16) 
    # hitch_minutes [T,S]: タスクtに対して、駅sに便乗合計所要（分）(int32)
    # paths [T,S]  dtype=object, 各要素は np.ndarray[int32]（時間順の便乗 task id 列）

    # --- Crew TD キーの要約 ---
    # C = 乗務員数
    # start_station_idx [C]: 各乗務員の開始/現在駅ID 
    # assignable_start_min [C]: 勤務可能になる最早時刻（分）
    # assignable_end_min [C]: 勤務規程上、勤務開始可能な遅最時刻（分）
    # end_station_ids_rag [C]: dtype=object 各乗務員の勤務終了可能駅IDの 配列

    def generate_raw_data(self):

        encoding = load_yaml(self.encoding_yaml)
        contraints = load_yaml(self.constraints_yaml)
        num_stations = len(encoding["station_id"])

        # 1) tt 生成
        tt ,train_num= generate_timetable(
            station_yaml_path=self.station_yaml,
            train_yaml_path=self.train_yaml,
            encoding=encoding,
            seed=None,
        )

        # 2) round
        round_arrays = build_round_cache(
            dep_time=tt["depart_time"],
            train_id=tt["train_id"],
        )
        
        # 3) cache（task×station）
        
        task_station_cache= build_task_station_cache(
            dep_station=tt["depart_station"],
            dep_time=tt["depart_time"],
            arr_station=tt["arrive_station"],
            arr_time=tt["arrive_time"],
            post_hitch_ready_min=int(contraints["post_hitch_ready_min"]),
            num_stations=num_stations,
            max_hops=int(contraints["max_hops"]),
            window_min=int(contraints["window_min"]),
        )
        crew = generate_initial_crew(self.crew_yaml,encoding)

        # 4) meta
        meta: Dict[str, Any] = dict(
            num_stations=num_stations,
            num_trains=train_num,
        )
        return tt,round_arrays,task_station_cache,crew,meta


from pathlib import Path
import numpy as np
import torch

def save_npz(path, td):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    npz_dict = {}
    for k, v in td.items():
        if isinstance(v, torch.Tensor):
            npv = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            npv = v
        elif isinstance(v, (list, tuple)):
            npv = np.array(v)
        elif isinstance(v, (int, float, bool, np.number)):
            npv = np.array(v)
        else:
            raise ValueError(f"Unsupported type in TensorDict: {type(v)} for key {k}")
        npz_dict[k] = npv

    np.savez_compressed(str(path), **npz_dict)


def load_npz(path) -> TensorDict:
    """
    .npz ファイルを TensorDict に復元します。
    - ndarray は torch.Tensor に変換
    - 0次元 ndarray は Python スカラに変換
    """
    td = {}
    npz = np.load(path, allow_pickle=True)  # allow_pickle=True は object dtype の場合に必要
    for k, v in npz.items():
        if isinstance(v, np.ndarray):
            if v.shape == ():  # 0次元配列 → Python スカラ
                td[k] = v.item()
            else:
                try:
                    td[k] = torch.from_numpy(v)
                except Exception:
                    td[k] = v  # torch に変換できない場合は numpy のまま残す
        else:
            td[k] = v
    return td

