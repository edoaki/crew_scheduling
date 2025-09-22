# 参考: model_stracture.txt に沿った雛形（CrewAREnv / AgentHandler）
# :contentReference[oaicite:0]{index=0}

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
import numpy as np

# state.py 側で定義されている想定の型（なければ各自用意して下さい）
# - StaticBundle: 問題インスタンスごとの静的情報束（generator出力＋各種キャッシュ）
# - DynamicState: ステップで更新される可変情報
# - WindowIndex: 現ラウンドのローカルID集合（可変長ビューのため）
# - ObservationView: モデル入力に渡す観測（可変長インデックスとマスク）
from .state import StaticBundle, DynamicState, WindowIndex,StaticObs,DynamicObs, ActionMask,RoundPairBias
from utils.io import Assignment
from rl_env.env_core import get_action_mask,build_pair_info,build_window_index

class CrewAREnv:
    """
    Gymを使わない自前RL環境。
    - reset(static) で問題インスタンス固有の静的情報を受け取り、動的状態を初期化
    - step(action) で「現ラウンドの割当て」を適用し、状態・報酬・次観測を返す
    - get_action_mask() で各種マスク（基礎可否・休憩・終電）を合成した行動可否を返す



    ##### 実装していないこと #####
    - 割当失敗時のペナルティ（現状は割当失敗を許さない前提）
    - 乗務員の緊急追加
    - 乗り込み時間 (board_min) の考慮（現状は0分想定）
    """

    def __init__(self,constraints) -> None:
        """
        ここでは環境の構成パラメータ（休憩最小時間・移動バッファ・罰則重み等）を保持する。
        - self.static: StaticBundle | None を後でresetでセット
        - self.dyn: DynamicState | None を後でresetで初期化
        - self.config: 各種閾値や係数、乱数器等
        """

        self.static: Optional[StaticBundle] = None
        self.dyn: Optional[DynamicState] = None

        self.constraints = constraints
        self.duty_max_min = constraints['duty_max_min']
        self.continuous_drive_max_min = constraints['continuous_drive_max_min']
        self.break_min = constraints['break_min']
        self.dispatch_prep_min = constraints['dispatch_prep_min']
        self.stabling_post_min = constraints['stabling_post_min']

        # === 設定　後でconfig.yaml等から読み込むようにする ===
        self.task_lookahead_min = 60  # タスクウィンドウ先読み幅（分）

    def reset(self, td: Dict):
        """
        入力: 問題インスタンスごとの静的情報（generatorが作成）
        """
        self.static  = StaticBundle.from_generator_dict(td)
        self.dyn = DynamicState.init_from_static(self.static,self.continuous_drive_max_min)
       
        window = build_window_index(self.static,self.dyn,self.task_lookahead_min)

        static_obs = StaticObs.obs_from_static(self.static)
        dyn_obs = DynamicObs.obs_from_dynamic(self.static,self.dyn, window)
        mask = get_action_mask(self.static,self.dyn,dyn_obs,self.continuous_drive_max_min)
        pair_info = build_pair_info(dyn_obs, mask,self.static,self.dyn)

        return static_obs,dyn_obs,mask,pair_info

    def step(self,assignment:Assignment):
        assert self.static is not None and self.dyn is not None, "reset() 実行前です。"

        
        # 状態更新: 割当結果を self.dyn に反映
        self._assignments_update_state(self.dyn,assignment)
        # 割り当てができなかった場合の処理も必要 （未実装）

        # 報酬計算
        reward = self._compute_reward(assignment)

         # 終了判定
        done = self._is_done()

        # ラウンド前進  
        next_round_time = self._advance_round()

        # 次ラウンド開始時の自動休憩反映,退勤判定
        self._auto_apply_crew(next_round_time)

        # 次の観測生成
        next_window = build_window_index(self.static,self.dyn,self.task_lookahead_min)
        dyn_next_obs = DynamicObs.obs_from_dynamic(self.static,self.dyn, next_window)
        mask = get_action_mask(self.static,self.dyn,dyn_next_obs,self.continuous_drive_max_min)
        pair_info = build_pair_info(dyn_next_obs, mask,self.static,self.dyn)
        
        info: Dict[str, Any] = {}
        return dyn_next_obs,mask,pair_info, float(reward), bool(done), info

    # ============== 休憩・退勤判定 ==============
    def _auto_apply_crew(self,round_time) -> None:
        """
        退勤・休憩の自動適用
        前提:
        - self.dyn 以下に各クルーのステート（tensor）があること
        - self.duty_max_min, self.continuous_drive_max_min, self.break_min はスカラー（int）
        """
        # 参照しやすい別名
        rt = round_time  # 現在のラウンド時刻 [min]
        # work_start = self.dyn.crew_work_start_min       # [C] int32
        # consec_work_min = self.dyn.crew_consec_work_min # [C] int32 (0で連続稼働なし)
        # consec_start = self.dyn.crew_consec_start_min   # [C] int32
        # ready_time = self.dyn.crew_ready_time                # [C] int32
        signoff_limit = self.static.crew_signoff_limit_min 

        idtype = self.dyn.crew_ready_time.dtype

        duty_max = torch.as_tensor(self.duty_max_min, dtype=idtype)
        cont_max = torch.as_tensor(self.continuous_drive_max_min, dtype=idtype)
        brk = torch.as_tensor(self.break_min, dtype=idtype)

        if not torch.is_tensor(rt):
            rt = torch.as_tensor(rt, dtype=idtype)

        # 退勤判定（duty超過とサインオフ超過を一括適用）
        over_duty    = self.dyn.crew_on_duty & ((rt - self.dyn.crew_work_start_min) > duty_max)
        over_signoff = self.dyn.crew_on_duty & (rt > signoff_limit)
        over_any     = over_duty | over_signoff

        # finished は累積、on_duty は外す
        self.dyn.crew_finished = self.dyn.crew_finished | over_any
        self.dyn.crew_on_duty  = self.dyn.crew_on_duty  & (~over_any)


        # 未着任のまま assignable_end_min を超過したクルーを終了扱いにする
        timeout_never_started = (~self.dyn.crew_on_duty) & (rt > self.static.assignable_end_min)
        self.dyn.crew_finished = self.dyn.crew_finished | timeout_never_started


        # 休憩判定の前提: 連続稼働中（= 0ではない）
        has_consec = self.dyn.crew_consec_start_min != -1

        # 連続稼働の休憩判定: (round_time - crew_consec_start_min) > continuous_drive_max_min
        need_break_consec = has_consec & ((rt - self.dyn.crew_consec_start_min) > cont_max)

        # 放置の休憩判定: round_time > ready_time + break_min
        need_break_idle = has_consec & (rt > (self.dyn.crew_ready_time + brk))
        # print(rt," ",self.dyn.crew_ready_time," ",brk)
        need_break = need_break_consec | need_break_idle
        # print("need_break",need_break)
        if need_break.any():
            # ready_time を break 分だけ後ろ倒し
            self.dyn.crew_ready_time = torch.where(need_break, self.dyn.crew_ready_time + brk, self.dyn.crew_ready_time)

            # 連続稼働リセット
            self.dyn.crew_consec_work_min = torch.where(
                need_break, torch.zeros_like(self.dyn.crew_consec_work_min), self.dyn.crew_consec_work_min
            )
            # -1にする
            self.dyn.crew_consec_start_min = torch.where(
                need_break, torch.full_like(self.dyn.crew_consec_start_min, -1), self.dyn.crew_consec_start_min
            )
            
            # 休憩後の連続稼働可能時間を満タンに
            self.dyn.crew_rest_remaining = torch.where(
                need_break, torch.full_like(self.dyn.crew_rest_remaining, cont_max), self.dyn.crew_rest_remaining
            )

        # on_duty かつstart_minが -1でない場合、crew_consec_work_minをround_time-consec_startに更新
        need_update = self.dyn.crew_on_duty & (self.dyn.crew_consec_start_min != -1)

        if need_update.any():
            self.dyn.crew_consec_work_min = torch.where(
                need_update, rt - self.dyn.crew_consec_start_min, self.dyn.crew_consec_work_min
            )
            self.dyn.crew_rest_remaining = torch.where(
                need_update, cont_max - self.dyn.crew_consec_work_min, self.dyn.crew_rest_remaining
            )

    def _assignments_update_state(self,dyn, assignment) -> None:
        """
        タスク割当てに基づき DynamicState を更新する。
        - now_round / now_round_time はここでは更新しない（advance_round側）。
        - owner は扱わず、task_assign のみ設定する。
        - crew_ready_time は担当タスクの arrive_time に更新する。
        - 作業量の定義は「タスク所要時間 = arrive_time - depart_time」。
        - on_duty が False の場合のみ、当日勤務開始のセット（on_duty=True, crew_duty_start_min設定）。
     
        """

        # print("= _assignments_update_state =")
        # print("assignment",assignment)

        if assignment.unassigned_task_ids.numel() > 0:
            print(f"[warn] 割り当てに失敗したタスク: {assignment.unassigned_task_ids.tolist()}")

            # must_be_by_minを表示する
            # for task_id in assignment.unassigned_task_ids.tolist():
            #     mbm_row = self.static.must_be_by_min[task_id]
                # print(f"  task {task_id} must_be_by_min: {mbm_row.tolist()}")
                # print(self.dyn.crew_ready_time)
                # print(" ",self.static.depart_station[task_id],"->",self.static.arrive_station[task_id]," ",self.static.depart_time[task_id],"->",self.static.arrive_time[task_id])

        valid_pairs = assignment.pairs[assignment.pairs[:, 0] != -1] 

        # Static 情報（名称差異に対応したフォールバック）
        depart_time_arr   = self.static.depart_time
        arrive_time_arr   = self.static.arrive_time
        arrive_station_arr= self.static.arrive_station
        train_id_arr      = self.static.train_id

        for crew_id, task_id in valid_pairs.tolist():
            print(f"Assign crew {crew_id} to task {task_id}")
            # タスク側: 担当クルーIDを設定
            self.dyn.task_assign[task_id] = crew_id

            # 時刻・駅・列車IDを取得
            depart_time = int(depart_time_arr[task_id].item())
            arrive_time = int(arrive_time_arr[task_id].item())
            arrive_station = int(arrive_station_arr[task_id].item())
            # train_id は long 前提だが item() で int 化
            train_id = int(train_id_arr[task_id].item())

            # on_duty がまだ False なら、当日勤務開始をセット
            if not bool(dyn.crew_on_duty[crew_id]):
                dyn.crew_on_duty[crew_id] = True
                dyn.crew_work_start_min[crew_id] = torch.tensor(depart_time, dtype=torch.int32)
                # 連続勤務ブロックの開始も未設定ならセット
                dyn.crew_consec_start_min[crew_id] = torch.tensor(depart_time, dtype=torch.int32)

            # 常時更新項目
            # 位置（到着駅）
            dyn.crew_station[crew_id] = torch.tensor(arrive_station, dtype=dyn.crew_station.dtype)
            # 次に割当可能になる時刻 = 到着時刻
            dyn.crew_ready_time[crew_id] = torch.tensor(arrive_time, dtype=torch.int32)

            mbm_row = self.static.must_be_by_min[task_id]
            # print(f"  task {task_id} must_be_by_min: {mbm_row.tolist()}")
            # print("pair ", crew_id, "->", task_id)
            # print(" ",self.static.depart_station[task_id],"->",self.static.arrive_station[task_id]," ",self.static.depart_time[task_id],"->",self.static.arrive_time[task_id])
            # print("crew ready ",dyn.crew_ready_time)
            # print("remaining ",dyn.crew_rest_remaining[crew_id])
            # print("consec ",dyn.crew_consec_work_min[crew_id])
            # print("pair " ,dyn.crew_ready_time[crew_id])

            # crew_consec_work_minが0の場合、休憩していたことになるので、今回の勤務のdepart_timeを開始時刻にconsec_start_minを合わせる
            if dyn.crew_consec_start_min[crew_id] == -1:
                dyn.crew_consec_start_min[crew_id] = torch.tensor(depart_time, dtype=torch.int32)
            
            dyn.crew_total_work_min[crew_id] = arrive_time - int(dyn.crew_work_start_min[crew_id].item())
            # taskのis_arrival_before_turnbackが1の場合は連続乗務は考慮しないので、last_crew_idは-1にする
            if self.static.is_arrival_before_turnback[task_id]:
                dyn.train_last_crew_id[train_id] = torch.tensor(-1, dtype=torch.long)
            else: # 連続勤務可能なため
                dyn.train_last_crew_id[train_id] = torch.tensor(crew_id, dtype=torch.long)

    def _advance_round(self) -> int:
        """
        ラウンドを1つ進める。
        - now_round を +1
        - now_time を round_time[now_round] に更新
        - 必要に応じて「3時間ウィンドウ」の再構成は次の観測生成時に行う
        """
        print("now ",self.dyn.now_round," round ",self.static.num_rounds)
        if self.dyn.now_round+1 >= self.static.num_rounds:
            self.dyn.now_round = self.static.num_rounds
            print("if 100000 done by round end")
            # ここで done を立て、時刻は現在ラウンドのものを維持
            return 100000  # ダミー値
    
        self.dyn.now_round += 1
        self.dyn.now_round_time = int(self.static.round_time[self.dyn.now_round].item())
        return self.dyn.now_round_time

    def _is_done(self) -> bool:
        assert self.static is not None and self.dyn is not None
        # 現ラウンド（0始まり）が最終ラウンド（num_rounds-1）に到達していれば終了
        if self.dyn.now_round >= self.static.num_rounds:
            print("done by round end")
            return True
        return False

    # 報酬

    def _compute_reward(self, assignments:Assignment) -> float:
        """
        今回ラウンドで発生した増分コスト/ペナルティ/ボーナスから報酬を算出して返す。
        例:
          - 移動/遅延/違反ペナルティの合成
          - 充足率や平準化のボーナス
        """
        # ここで報酬計算（未実装）
        return 0.0
