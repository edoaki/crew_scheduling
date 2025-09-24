from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np

# ==============================
# 基本方針
# - 時間は分（long）
# - ID/インデックスはtorch.long（int64）
# - boolはtorch.bool
# ==============================

# -------- generator出力（静的）を保持する束 --------
@dataclass
class StaticBundle:
    # タスク（全日・グローバル）
    train_id: torch.Tensor           # [T] long
    service: torch.Tensor            # [T] long
    direction: torch.Tensor          # [T] long
    depart_station: torch.Tensor     # [T] long (駅ID)
    arrive_station: torch.Tensor     # [T] long
    depart_time: torch.Tensor        # [T] long (分)
    arrive_time: torch.Tensor        # [T] long (分)
    is_dispatch_task: torch.Tensor   # [T] bool
    is_depart_from_turnback: torch.Tensor # [T] bool
    is_arrival_before_turnback: torch.Tensor # [T] bool
    is_stabling_at_arrival: torch.Tensor    # [T] bool
    next_event_time_from_depart: torch.Tensor # [T] long

    # ラウンド
    round_first_task_id: torch.Tensor  # [R] long（各ラウンド先頭taskのglobal ID）
    round_task_to_round: torch.Tensor  # [T] long（各taskが属するラウンド）
    round_time: torch.Tensor           # [R] long（各ラウンドの代表時刻など）

    # 乗務可否キャッシュ（タスク×駅）
    station_ids: torch.Tensor       # [S] long
    task_ids: torch.Tensor          # [T] long（0..T-1想定）
    must_be_by_min: torch.Tensor    # [T, S] long（その駅発なら「◯分までに居る必要」）
    is_hitch: torch.Tensor          # [T, S] bool（その駅から当該タスクへ“拾える”か）
    hops: torch.Tensor              # [T, S] long
    hitch_minutes: torch.Tensor     # [T, S] long（その駅から当該タスクまで必要な移動分）
    


    # 乗務員の不変属性
    start_station_idx: torch.Tensor       # [C] long
    assignable_start_min: torch.Tensor    # [C] long
    assignable_end_min: torch.Tensor      # [C] long
    crew_slot_label: torch.Tensor         # [C] int8  (am=0, pm=1,追加人員は2)
    crew_signoff_limit_min: torch.Tensor  # [C] long   

    # 便利サイズ
    num_tasks: int
    num_crews: int
    num_stations: int
    num_rounds: int
    num_trains: int 

    @staticmethod
    def from_generator_dict(g: Dict) -> "StaticBundle":
        def t(a, dtype=None, as_long=False, squeeze=True):
            x = torch.from_numpy(np.asarray(a))
            if squeeze and x.ndim >= 1 and x.shape[0] == 1:
                x = x.squeeze(0)
            if dtype is not None:
                x = x.to(dtype)
            if as_long:
                x = x.to(torch.long)
            return x

        # 基本配列
        train_id = t(g["train_id"], torch.long)
        service = t(g["service"], torch.long)
        direction = t(g["direction"], torch.long)
        depart_station = t(g["depart_station"], torch.long)
        arrive_station = t(g["arrive_station"], torch.long)
        depart_time = t(g["depart_time"], torch.long)   
        arrive_time = t(g["arrive_time"], torch.long)
        is_dispatch_task = t(g["is_dispatch_task"],torch.bool)
        is_depart_from_turnback = t(g["is_depart_from_turnback"],torch.bool)
        is_arrival_before_turnback = t(g["is_arrival_before_turnback"],torch.bool)
        is_stabling_at_arrival = t(g["is_stabling_at_arrival"],torch.bool)
        next_event_time_from_depart = t(g["next_event_time_from_depart"], torch.long)

        round_first_task_id = t(g["round_first_task_id"], torch.long)
        round_task_to_round = t(g["round_task_to_round"], torch.long)
        round_time = t(g["round_time"], torch.long)

        station_ids = t(g["station_ids"], torch.long)
        task_ids = t(g["task_ids"], torch.long)
        # [T, S]にreshape
        T = task_ids.numel()
        S = station_ids.numel()
        # タスクごとに駅数分の情報
        must_be_by_min = t(g["must_be_by_min"], torch.long).reshape(T, S)
        is_hitch = t(g["is_hitch"]).to(torch.bool).reshape(T, S)
        hops = t(g["hops"]).to(torch.int16).reshape(T, S)
        hitch_minutes = t(g["hitch_minutes"], torch.long).reshape(T, S)


        start_station_idx = t(g["start_station_idx"], torch.long)
        assignable_start_min = t(g["assignable_start_min"], torch.long)
        assignable_end_min = t(g["assignable_end_min"], torch.long)
        crew_slot_label = t(g["crew_slot_label"], torch.long)
        crew_signoff_limit_min = t(g["crew_signoff_limit_min"], torch.long)

        R = round_first_task_id.numel()
        C = start_station_idx.numel()

        num_trains = t(g["num_trains"], torch.long)
        
        # T タスク数 , C 乗務員数 , S 駅数 , R ラウンド数

        return StaticBundle(
            train_id=train_id, #
            service=service, # local/rapid が0/1
            direction=direction, #  up/down 0/1
            depart_station=depart_station, 
            arrive_station=arrive_station,  
            depart_time=depart_time, # [T] time int形式
            arrive_time=arrive_time, # [T] time
            is_dispatch_task=is_dispatch_task, # 
            is_depart_from_turnback=is_depart_from_turnback,
            is_arrival_before_turnback=is_arrival_before_turnback,
            is_stabling_at_arrival=is_stabling_at_arrival,
            next_event_time_from_depart=next_event_time_from_depart,
            round_first_task_id=round_first_task_id,
            round_task_to_round=round_task_to_round,
            round_time=round_time,
            station_ids=station_ids, task_ids=task_ids,
            must_be_by_min=must_be_by_min, # time int形式
            is_hitch=is_hitch,hops=hops, 
            hitch_minutes=hitch_minutes,
            start_station_idx=start_station_idx,
            assignable_start_min=assignable_start_min, 
            assignable_end_min=assignable_end_min,
            crew_slot_label=crew_slot_label,
            crew_signoff_limit_min=crew_signoff_limit_min,
            num_tasks=T, num_crews=C, num_stations=S, num_rounds=R,num_trains=num_trains
        )

# -------- 動的状態（毎ステップ更新） --------
@dataclass
class DynamicState:
    """エピソード進行中に更新される値（全て“グローバルID”基準）"""
    now_round: int                 # 現在ラウンド（0..R-1）
    now_round_time: int            # 現在ラウンドの時刻（分）

    # タスク側
    task_assign: torch.Tensor    # [T] int 担当した乗務員ID（未割当は -1）
    # 電車に直近で乗った乗務員のID（未乗車は -1）
    train_last_crew_id: torch.Tensor   

    # 乗務員側
    crew_station: torch.Tensor         # [C] long（現在駅ID）
    crew_ready_time: torch.Tensor      # [C] long（次に割当可能になる時刻）
    crew_on_duty: torch.Tensor         # [C] bool（現在勤務中か）
    crew_finished: torch.Tensor        # [C] bool（本日勤務終了済みか）   
    crew_rest_remaining: torch.Tensor  # [C] long（次休憩までの残り分）

    # 推奨：運用上あると便利な可変フィールド
    crew_consec_work_min: torch.Tensor     # [C] long（連続勤務分）
    crew_consec_start_min: torch.Tensor  # [C] long（連続勤務開始時刻。未開始は -1）
    crew_total_work_min: torch.Tensor      # [C] long（当日勤務累計分）
    crew_work_start_min: torch.Tensor      # [C] long（当日勤務開始時刻。未開始は -1）

    @staticmethod
    def init_from_static(b: StaticBundle,continuous_drive_max_min:int) -> "DynamicState":
        T, C = b.num_tasks, b.num_crews
        train_num = b.num_trains
        return DynamicState(
            now_round=0,
            now_round_time=int(b.round_time[0].item()),
            task_assign=torch.full((T,), -1, dtype=torch.long),  # 未割当=-1
            train_last_crew_id=torch.full((train_num,), -1, dtype=torch.long),
            crew_station=b.start_station_idx.clone(),
            crew_ready_time=b.assignable_start_min.clone(),
            crew_on_duty=torch.zeros(C, dtype=torch.bool),
            crew_finished=torch.zeros(C, dtype=torch.bool),
            crew_rest_remaining=torch.full((C,), continuous_drive_max_min, dtype=torch.long),
            crew_consec_work_min=torch.zeros(C, dtype=torch.long),
            crew_consec_start_min=torch.full((C,), -1, dtype=torch.long),
            crew_total_work_min=torch.zeros(C, dtype=torch.long),
            crew_work_start_min=torch.full((C,), -1, dtype=torch.long),
        )


# --- 観測（エピソード中に不変の情報）---
# 問題インスタンスで固定の値で全体の情報を持つ
@dataclass
class StaticObs:

    # メタ情報
    num_tasks: int # T
    num_crews: int # C
    num_rounds: int # R

    # ---- Task 静的属性（StaticBundle 由来：変わらない）---- 
    # task(行路)ごと情報
    service: torch.Tensor                  # [T]  long (0=local,1=rapid)
    direction: torch.Tensor                # [T]  long (0=up,1=down)
    depart_station: torch.Tensor           # [T]  long
    arrive_station: torch.Tensor           # [T]  long
    depart_time: torch.Tensor              # [T]  long (分)
    arrive_time: torch.Tensor              # [T]  long (分)
    is_dispatch_task: torch.Tensor         # [T]  bool
    is_depart_from_turnback: torch.Tensor  # [T]  bool
    is_arrival_before_turnback: torch.Tensor  # [T]  bool
    is_stabling_at_arrival: torch.Tensor    # [T]  bool
    round_task_to_round: torch.Tensor  # [T]  long（各taskが属するラウンド）

    # ---- Crew 静的属性 ----
    crew_start_station: torch.Tensor       # [C] long (駅ID 0~6)
    crew_assignable_start_min: torch.Tensor  # [C] long
    crew_assignable_end_min: torch.Tensor    # [C] long
    crew_slot_label: torch.Tensor         # [C] long (am=0, pm=1,追加人員は2)
    crew_signoff_limit_min: torch.Tensor  # [C] long

    @staticmethod
    def obs_from_static(static: StaticBundle) -> StaticObs:
        return StaticObs(
            service=static.service,
            direction=static.direction,
            depart_station=static.depart_station,
            arrive_station=static.arrive_station,
            depart_time=static.depart_time,
            arrive_time=static.arrive_time,
            is_dispatch_task=static.is_dispatch_task,
            is_depart_from_turnback=static.is_depart_from_turnback,
            is_arrival_before_turnback=static.is_arrival_before_turnback,
            is_stabling_at_arrival=static.is_stabling_at_arrival,
            round_task_to_round=static.round_task_to_round,
            crew_start_station=static.start_station_idx,
            crew_assignable_start_min=static.assignable_start_min,
            crew_assignable_end_min=static.assignable_end_min,
            crew_slot_label=static.crew_slot_label,
            crew_signoff_limit_min=static.crew_signoff_limit_min,
            num_tasks=static.num_tasks,
            num_crews=static.num_crews,
            num_rounds=static.num_rounds,
        )

# --- 観測（各ステップで変わる情報／静的は含めない）---
# window
@dataclass
class DynamicObs:
    # グローバル
    current_round_id: int                  # いま処理中のラウンドID
    local_task_ids: torch.Tensor      # [w_T] long  window内のタスクID
    local_crew_ids: torch.Tensor      # [w_C] long  window内の乗務員ID
    round_task_ids: torch.Tensor      # [R_t] long  今ラウンドのタスクID
    
    # Crew 動的
    crew_location_station: torch.Tensor    # [w_C] long  現在駅
    crew_ready_time: torch.Tensor          # [w_C] long 次に割当可能になる時刻
    crew_consec_work_min: torch.Tensor     # [w_C] long（連続勤務分）
    crew_rest_remaining: torch.Tensor      # [w_C] long 次休憩までの残り分
    crew_on_duty: torch.Tensor            # [w_C] bool（現在勤務中か）
    crew_duty_minutes: torch.Tensor        # [w_C] long 当日累計稼働

    @staticmethod
    def obs_from_dynamic(static: StaticBundle,dyn: DynamicState, window: WindowIndex) -> DynamicObs:


        return DynamicObs(
            current_round_id=int(dyn.now_round),
            local_task_ids=window.local_task_ids,
            local_crew_ids=window.local_crew_ids,
            round_task_ids=window.round_task_ids,

            crew_location_station=dyn.crew_station[window.local_crew_ids],
            crew_ready_time=dyn.crew_ready_time[window.local_crew_ids],
            crew_consec_work_min=dyn.crew_consec_work_min[window.local_crew_ids],
            crew_rest_remaining=dyn.crew_rest_remaining[window.local_crew_ids],
            crew_on_duty = dyn.crew_on_duty[window.local_crew_ids],
            crew_duty_minutes=dyn.crew_total_work_min[window.local_crew_ids],
        )
    @staticmethod
    def init_dummy(T:int,C:int)->"DynamicObs":
        return DynamicObs(
            current_round_id=0,
            local_task_ids=torch.zeros((T,),dtype=torch.long),
            local_crew_ids=torch.zeros((C,),dtype=torch.long),
            round_task_ids=torch.zeros((T,),dtype=torch.long),
            crew_location_station=torch.zeros((C,),dtype=torch.long),
            crew_ready_time=torch.zeros((C,),dtype=torch.long),
            crew_consec_work_min=torch.zeros((C,),dtype=torch.long),
            crew_rest_remaining=torch.zeros((C,),dtype=torch.long),
            crew_on_duty = torch.zeros((C,),dtype=torch.bool),
            crew_duty_minutes=torch.zeros((C,),dtype=torch.long),
        )
    
# -------- ラウンドのローカルビュー（観測切出しのための索引） --------
@dataclass
class WindowIndex:
    local_task_ids: torch.Tensor   # window内すべてのタスク (昇順)
    local_crew_ids: torch.Tensor   # window内すべてのクルー
    round_task_ids: torch.Tensor   # 今ラウンドのタスクだけ
    round_task_count: int   


@dataclass
class RoundPairBias:
    """
    行: local_crew_ids（今回ウィンドウのクルー）
    列: round_task_ids（今回ラウンドのタスク）
    - is_hitch:        便乗が必要かを表すフラグ（0=同駅で便乗なし/便乗不可, 1=便乗が必要で可行）
    - hitch_minutes:   便乗に要する分（不可や同駅は0）
    - is_continuous:   連続勤務（前担当クルー==このクルー）フラグ
    - rel_time:        on_duty==True: arrive_time - crew_ready_time
                       on_duty==False: hitch_minutes
    - on_duty:         クルーが勤務開始済みか（レジーム識別用、タスク次元へブロードキャスト済）
    """
    rows_crew_ids: torch.Tensor   # [C] long pad=-1
    cols_task_ids: torch.Tensor   # [T_round] long pad=-1
    is_hitch: torch.Tensor        # [C, T_round] bool
    hitch_minutes: torch.Tensor   # [C, T_round] long(>=0)
    is_continuous: torch.Tensor   # [C, T_round] bool
    rel_time: torch.Tensor        # [C, T_round] long
    on_duty: torch.Tensor         # [C, T_round] bool

    @staticmethod
    def init_dummy(C:int,T_round:int)->"RoundPairBias":
        return RoundPairBias(
            rows_crew_ids=torch.full((C,),-1,dtype=torch.long),
            cols_task_ids=torch.full((T_round,),-1,dtype=torch.long),
            is_hitch=torch.zeros((C,T_round),dtype=torch.bool),
            hitch_minutes=torch.zeros((C,T_round),dtype=torch.long),
            is_continuous=torch.zeros((C,T_round),dtype=torch.bool),
            rel_time=torch.zeros((C,T_round),dtype=torch.long),
            on_duty=torch.zeros((C,T_round),dtype=torch.bool),
        )

@dataclass
class ActionMask:
    """
    行: local_crew_ids（今回ウィンドウのクルー）
    列: round_task_ids（今回ラウンド内に検討するタスク）
    matrix: [C, T_round] の bool 行列（True = 割当可能）
    """
    rows_crew_ids: torch.Tensor   # shape [C], dtype=torch.long（DynamicObs.local_crew_ids）
    cols_task_ids: torch.Tensor   # shape [T_round], dtype=torch.long（DynamicObs.round_task_ids）
    matrix: torch.Tensor          # shape [C, T_round], dtype=torch.bool

    @staticmethod
    def init_dummy(C:int,T_round:int)->"ActionMask":
        return ActionMask(
            rows_crew_ids=torch.zeros((C,),dtype=torch.long),
            cols_task_ids=torch.zeros((T_round,),dtype=torch.long),
            matrix=torch.zeros((C,T_round),dtype=torch.bool),
        )