from rl_env.state import StaticBundle, DynamicState, DynamicObs, ActionMask, RoundPairBias, WindowIndex
import torch

def build_pair_info(dynamicobs: DynamicObs, mask: ActionMask, static: StaticBundle, dyn: DynamicState) -> RoundPairBias:
    """
    今回ウィンドウ内のクルー × 今ラウンドのタスクに対するペア特徴を構築する。
    形状はすべて [C, T_round]。不可セル（mask.matrix=False）は中立値で埋める。
    - is_hitch:       0=同駅で便乗不要 / 0=便乗不可 / 1=便乗が必要で可行
    - hitch_minutes:  便乗に必要な分（同駅/不可は0）
    - is_continuous:  連続勤務フラグ（前担当クルー==このクルー）
    - rel_time:       on_duty==True: arrive_time - crew_ready_time
                      on_duty==False: hitch_minutes
    - on_duty:        クルーの勤務開始フラグ（タスク次元へブロードキャスト）
    """
    crew_ids = dynamicobs.local_crew_ids.long()     # [C]
    task_ids = dynamicobs.round_task_ids.long()     # [T]
    C = int(crew_ids.shape[0])
    T = int(task_ids.shape[0])

    if C == 0 or T == 0:
        zeros_bool = torch.zeros((C, T), dtype=torch.bool)
        zeros_long = torch.zeros((C, T), dtype=torch.long)
        return RoundPairBias(
            rows_crew_ids=crew_ids,
            cols_task_ids=task_ids,
            is_hitch=zeros_bool,
            hitch_minutes=zeros_long,
            is_continuous=zeros_bool,
            rel_time=zeros_long,
            on_duty=zeros_bool,
        )

    # --- 便乗キャッシュ（タスク×駅）から [C,T] へ射影 ---
    # [T, S] を round タスクに絞る
    is_hitch_TS = static.is_hitch.index_select(0, task_ids).to(torch.bool)           # [T, S]
    hitch_min_TS = static.hitch_minutes.index_select(0, task_ids).to(torch.long)     # [T, S] (>=0想定)

    # クルー現在駅を列参照して [T, C] → 転置で [C, T]
    crew_station = dynamicobs.crew_location_station.long()                           # [C]
    is_hitch_CT = is_hitch_TS[:, crew_station].transpose(0, 1).contiguous()          # [C, T]
    hitch_min_CT = hitch_min_TS[:, crew_station].transpose(0, 1).contiguous()        # [C, T]

    # --- 連続勤務フラグ（前担当クルー==このクルー）---
    # taskごとの train_id → 直近乗務クルーID と比較
    train_ids_T = static.train_id.index_select(0, task_ids).long()                   # [T]
    last_crew_for_train_T = dyn.train_last_crew_id.index_select(0, train_ids_T)      # [T] （未乗車は -1）
    is_continuous_CT = (last_crew_for_train_T.unsqueeze(0) == crew_ids.view(C, 1))   # [C, T], bool

    # --- on_duty のブロードキャスト ---
    on_duty_C = dynamicobs.crew_on_duty.to(torch.bool)                                # [C]
    on_duty_CT = on_duty_C.unsqueeze(1).expand(C, T).contiguous()                     # [C, T]

    # --- rel_time の構築 ---
    # on_duty==True: arrive_time - crew_ready_time
    depart_T = static.depart_time.index_select(0, task_ids).to(torch.long)            # [T]
    crew_ready_C = dynamicobs.crew_ready_time.to(torch.long)                           # [C]
    rel_time_on_CT = depart_T.unsqueeze(0) - crew_ready_C.unsqueeze(1)                # [C, T]

    # on_duty==False: hitch_minutes
    rel_time_off_CT = hitch_min_CT                                                    # [C, T]

    rel_time_CT = torch.where(on_duty_CT, rel_time_on_CT, rel_time_off_CT)            # [C, T]

    # --- 行動マスク適用（不可セルは中立値に固定）---
    # 中立値: is_hitch=False, hitch_minutes=0, is_continuous=False, rel_time=0
    if mask is not None and hasattr(mask, "matrix"):
        feasible = mask.matrix.to(torch.bool)                                         # [C, T]
    else:
        feasible = torch.ones((C, T), dtype=torch.bool, device=crew_ids.device)

    is_hitch_CT = is_hitch_CT & feasible
    hitch_min_CT = hitch_min_CT * feasible.to(torch.long)
    is_continuous_CT = is_continuous_CT & feasible
    rel_time_CT = rel_time_CT * feasible.to(torch.long)
    # on_duty はペアの可否に依らず状態を示すのでそのまま（必要なら可否で0にしたい場合は下記を使用）
    # on_duty_CT = on_duty_CT & feasible

    return RoundPairBias(
        rows_crew_ids=crew_ids,
        cols_task_ids=task_ids,
        is_hitch=is_hitch_CT,
        hitch_minutes=hitch_min_CT,
        is_continuous=is_continuous_CT,
        rel_time=rel_time_CT,
        on_duty=on_duty_CT,
    )

def get_action_mask(static:StaticBundle,dyn:DynamicState, dynamicobs: DynamicObs,continuous_drive_max_min:int) -> ActionMask:
    """
    現ラウンドのタスク集合と乗務員集合に対する行動可否マスクを返す。
    合成順:
        (a) 基礎可否（ready/キャッシュ/未割当/勤務時間帯）
        (b) 休憩制限プレッシャ（次の休憩制限に抵触見込みの組を不可化）
    戻り値: ActionMask（乗務員×タスクのbool行列）
    """
    # (a) 基礎可否
    base_mask = compute_feasible_mask_base(static,dynamicobs=dynamicobs)  # [C, T], bool

    # (b) 休憩プレッシャ（同形の bool 行列を返す想定）
    mask = apply_rest_pressure_mask(static,dyn=dyn,dynamicobs=dynamicobs,limit = continuous_drive_max_min,base_mask=base_mask)                                               # [C, T]

    return ActionMask(
        rows_crew_ids=dynamicobs.local_crew_ids.to(dtype=torch.long),
        cols_task_ids=dynamicobs.round_task_ids.to(dtype=torch.long),
        matrix=mask.to(dtype=torch.bool),
    )

def compute_feasible_mask_base(static: StaticBundle, dynamicobs: DynamicObs):
    """
    役割: must_be_by_min に基づき、クルー×タスクの到達可否を判定する。
    期待する前提:
        - dynamicobs.local_crew_ids: [C]
        - dynamicobs.round_task_ids: [T]
        - dynamicobs.crew_station_id: [C]  各クルーの現在駅ID
        - dynamicobs.crew_ready_time: [C]  各クルーが次に出られる時刻（分など）
        - static.must_be_by_min: [N_task, N_station]
            値の意味: 「駅 s にいる状態からタスク t に間に合うための、
                    クルーの ready_time の上限（<= なら間に合う）」
                    到達不可能は負値（例: -1）で表す想定
    戻り値:
        - base_mask: torch.bool [C, T]
    """

    crew_ids = dynamicobs.local_crew_ids    # [C]
    task_ids = dynamicobs.round_task_ids  # [T]
    C = crew_ids.shape[0]
    T = task_ids.shape[0]
    # ラウンド内タスク行だけ抽出: [T, N_station]

    if C == 0 or T == 0:
        return torch.zeros((C, T), dtype=torch.bool)

    # クルー側の状態（local_crew_ids と同じ並びで保持している前提）
    crew_station = dynamicobs.crew_location_station    # [C]
    crew_ready_time = dynamicobs.crew_ready_time # [C]

    # ラウンド内タスク行だけ抽出: [T, N_station]
    mbm_round = static.must_be_by_min.index_select(0, task_ids)    # [T, S]

    # 各クルーの station 列だけを抜き出す: mbm_round[:, crew_station] -> [T, C] を転置
    # thresholds[c, t] = must_be_by_min[ task_ids[t], crew_station[c] ]
    thresholds = mbm_round[:, crew_station].transpose(0, 1).contiguous()             # [C, T]

    # 到達不可能（負値想定）は False にする
    reachable = thresholds >= 0                                                      # [C, T]

    # ready_time が閾値以下なら間に合う
    time_ok = (crew_ready_time.view(C, 1) <= thresholds)                             # [C, T]

    base_mask = reachable & time_ok                                                  # [C, T]
    return base_mask

def apply_rest_pressure_mask(static:StaticBundle ,dyn:DynamicState,dynamicobs: DynamicObs,limit :int, base_mask) -> torch.Tensor:
    """
    役割: 勤務中の乗務員について「次の休憩制限（連続乗務上限）」に抵触見込みの割当候補を不可化する。
    
    - base_mask が与えられた場合はその位置のみ評価し、それ以外は False のまま残す。
    戻り値: torch.bool [C, T]
    """

    crew_ids = dynamicobs.local_crew_ids.long()   # [C]
    task_ids = dynamicobs.round_task_ids.long()   # [T]
    C, T = crew_ids.shape[0], task_ids.shape[0]

    if C == 0 or T == 0:
        return torch.zeros((C, T), dtype=torch.bool)

    # 入力
    arrive_time_round = static.arrive_time.index_select(0, task_ids).long()  # [T]

    # base_mask が無ければ全体を対象にする
    if base_mask is None:
        base_mask = torch.ones((C, T), dtype=torch.bool)

    consec_start_min = dyn.crew_consec_start_min[crew_ids].long()  # [C]
    new_consec = arrive_time_round.view(1, T) - consec_start_min.view(C, 1)  # [C, T]
    new_consec = torch.where(
        (consec_start_min.view(C, 1) == -1),
        torch.zeros_like(new_consec),
        new_consec
    )
    # print("new_consec", new_consec)
    
    ok = base_mask & (new_consec <= limit) 
    return ok

# ============== 観測生成・索引関連 ==============
def build_window_index(s :StaticBundle,d:DynamicState,task_lookahead:int) -> WindowIndex:
    """
    全タスク（グローバル）を depart_time 昇順でみなして、
    - local_task_ids: ラウンド先頭より前のタスクは除外し、基準時刻+lookahead を超えたら打ち切り
    - local_crew_ids: 除外条件が (crew_finished==True) OR (crew_ready_time > cutoff_time)
    - round_task_ids: 現ラウンドのタスク範囲のみ（round_first_task_id から決定）
    """

    r = int(d.now_round)  
    # print("build_window_index for round ", r)
    if r >= int(s.num_rounds):
        return WindowIndex(
            local_task_ids=torch.empty((0,), dtype=torch.long),
            local_crew_ids=torch.empty((0,), dtype=torch.long),
            round_task_ids=torch.empty((0,), dtype=torch.long),
            round_task_count=0,
        )
    round_start = int(s.round_first_task_id[r].item())
    if r + 1 < int(s.num_rounds):
        round_end = int(s.round_first_task_id[r + 1].item())
    else:
        round_end = int(s.num_tasks)

    cutoff_time = int(s.round_time[r].item()) + int(task_lookahead)

    # round_start 以降だけを対象に searchsorted
    dep_slice = s.depart_time[round_start:]
    offset = round_start
    local_end_rel = int(torch.searchsorted(dep_slice, torch.tensor(cutoff_time, dtype=dep_slice.dtype), right=True).item())
    end_idx = min(offset + local_end_rel, int(s.num_tasks))

    local_task_ids = torch.arange(round_start, end_idx, dtype=torch.long)

    # クルーのフィルタ：除外が OR 条件
    # 使うのは「採用側のマスク」なので否定をとる
    # keep = NOT( finished OR ready_time > cutoff )
    all_crew = torch.arange(int(s.num_crews), dtype=torch.long)
    exclude_mask = d.crew_finished | (d.crew_ready_time > cutoff_time)
    keep_mask = ~exclude_mask
    local_crew_ids = all_crew[keep_mask]

    # ラウンド内タスクIDは round_first_task_id から決定（当該ラウンド範囲のみ）
    round_task_ids = torch.arange(round_start, round_end, dtype=torch.long)
    round_task_count = int(round_end - round_start)


    return WindowIndex(
        local_task_ids=local_task_ids,
        local_crew_ids=local_crew_ids,
        round_task_ids=round_task_ids,
        round_task_count=round_task_count,
    )