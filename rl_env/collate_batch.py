from typing import List, Optional, Dict, Any
import torch
from rl_env.state import StaticObs, DynamicObs,ActionMask,RoundPairBias

PADDING_IDX = 0  # Embedding の padding_idx=0 を想定

def _pad1d(x: torch.Tensor, max_len: int, pad_value) -> torch.Tensor:
    L = int(x.shape[0])
    if L == max_len:
        return x
    if L > max_len:
        return x[:max_len]
    pad = torch.full((max_len - L,), pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)

def collate_static_obs(
    statics: List[StaticObs],
    device: Optional[torch.device] = None,
    return_aux: bool = True,
    pad_idx: int = PADDING_IDX,
) -> Dict[str, Any]:
    assert len(statics) > 0, "batch が空です"

    # device 決定
    if device is None:
        if hasattr(statics[0], "service") and isinstance(statics[0].service, torch.Tensor):
            device = statics[0].service.device
        else:
            device = torch.device("cpu")

    # 各サンプルのタスク長（service を基準に）
    T_list = [int(getattr(s, "service").shape[0]) for s in statics]
    T_max = max(T_list)
    B = len(statics)

    # パディングマスク（True=実データ, False=PAD）
    task_mask_list = []
    for T in T_list:
        ones = torch.ones(T, dtype=torch.bool, device=device)
        task_mask_list.append(_pad1d(ones, T_max, False))
    task_mask = torch.stack(task_mask_list, dim=0)  # [B, T_max]

    # 共通スタック関数
    def stack_task(attr: str, pad_value) -> torch.Tensor:
        xs = []
        for s in statics:
            x = getattr(s, attr)
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            x = x.to(device)
            xs.append(_pad1d(x, T_max, pad_value))
        return torch.stack(xs, dim=0)  # [B, T_max]

    # 既知のフィールド
    tasks: Dict[str, torch.Tensor] = {
        # カテゴリ（後で+1シフトして padding_idx=0 を確保）
        "service":         stack_task("service", pad_idx),
        "direction":       stack_task("direction", pad_idx),
        "depart_station":  stack_task("depart_station", pad_idx),
        "arrive_station":  stack_task("arrive_station", pad_idx),
        # 数値（時間は0でパディング、maskで無視）
        "depart_time":     stack_task("depart_time", 0),
        "arrive_time":     stack_task("arrive_time", 0),
    }

    # is_ 系の bool を自動収集（T次元の1Dテンソルのみ）
    for name in dir(statics[0]):
        if not name.startswith("is_"):
            continue
        v0 = getattr(statics[0], name)
        if isinstance(v0, torch.Tensor) and v0.dim() == 1 and v0.shape[0] == T_list[0]:
            tasks[name] = stack_task(name, False).to(torch.bool)

    # ---- カテゴリを+1シフト（実データ位置のみ） ----
    for key in ("service", "direction", "depart_station", "arrive_station"):
        if key in tasks:
            t = tasks[key]
            t = t.clone()
            t[task_mask] = t[task_mask] + 1  # 実データのみ +1
            t[~task_mask] = 0                # PADは0のまま
            tasks[key] = t.to(torch.long)

    # 整数系は long に寄せておく
    for key in ("depart_time", "arrive_time"):
        if key in tasks:
            tasks[key] = tasks[key].to(torch.long)
    for key in ("service", "direction", "depart_station", "arrive_station", "train_id"):
        if key in tasks:
            tasks[key] = tasks[key].to(torch.long)

    out: Dict[str, Any] = {
        "tasks": tasks,  # 各 [B, T_max]
        "pad_masks": {
            "tasks": task_mask,  # [B, T_max] True=実データ, False=PAD
        },
    }

    if return_aux:
        aux = {
            "task_len": torch.tensor(T_list, dtype=torch.long, device=device),
            "T_max": torch.tensor([T_max], dtype=torch.long, device=device),
        }
        out["aux"] = aux

    return out

def collate_dynamic_obs(
    dyns: List[DynamicObs],
    device: Optional[torch.device],
    done : Optional[torch.Tensor],
    return_aux: bool = False,
) -> Dict[str, Any]:
    """
    DynamicObs をバッチ化してパディングする。
    - タスク側は DynamicObs の順序を維持して [B, T_max] に揃える
      * local_task_ids: long（PAD=PADDING_IDX）
      * round_bool    : bool（今回ラウンドのタスクかどうか）
    - クルー側は [B, C_max] に揃える
      * crew_location_station: long（PAD=PADDING_IDX）
      * crew_on_duty, crew_finished: bool（PAD=False）
      * 時間系: crew_ready_time, crew_consec_work_min, crew_rest_remaining, crew_duty_minutes: long（PAD=0）
    - マスク:
      * task_mask: [B, T_max]（True=実データ, False=PAD）
      * crew_mask: [B, C_max]（True=実データ, False=PAD）
    - 0 長（T_max=0 or C_max=0）も許容する。
    """
    if device is None:
        device = torch.device("cpu")

    B = len(dyns)

    # --- 長さの収集 ---
    T_list = [int(x.local_task_ids.numel()) for x in dyns]
    C_list = [int(x.local_crew_ids.numel()) for x in dyns]
    T_max = int(max(T_list) if B > 0 else 0)
    C_max = int(max(C_list) if B > 0 else 0)

    def _pad1d(x: torch.Tensor, max_len: int, pad_value) -> torch.Tensor:
        L = int(x.shape[0])
        if max_len == 0:
            # 0 長のときは空テンソルを返す
            return torch.empty((0,), dtype=x.dtype, device=device)
        if L == max_len:
            return x
        if L > max_len:
            return x[:max_len]
        pad = torch.full((max_len - L,), pad_value, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=0)

    # --- タスク側のパディング ---
    if T_max == 0:
        local_task_ids = torch.empty((B, 0), dtype=torch.long, device=device)
        round_bool = torch.empty((B, 0), dtype=torch.bool, device=device)
        task_mask = torch.empty((B, 0), dtype=torch.bool, device=device)
    else:
        local_task_ids_list = []
        round_bool_list = []
        task_mask_list = []
        for x in dyns:
            lt = x.local_task_ids.to(torch.long).to(device)
            # 今回ラウンドのタスクに該当するか（順序は local_task_ids のまま）
            if x.round_task_ids.numel() == 0 or lt.numel() == 0:
                rb = torch.zeros((lt.numel(),), dtype=torch.bool, device=lt.device)
            else:
                rb = torch.isin(lt, x.round_task_ids.to(lt.device))
            lt_pad = _pad1d(lt, T_max, PADDING_IDX)
            rb_pad = _pad1d(rb, T_max, False)
            mask = torch.zeros((T_max,), dtype=torch.bool, device=device)
            mask[: lt.numel()] = True

            local_task_ids_list.append(lt_pad)
            round_bool_list.append(rb_pad)
            task_mask_list.append(mask)

        local_task_ids = torch.stack(local_task_ids_list, dim=0)
        round_bool = torch.stack(round_bool_list, dim=0)
        task_mask = torch.stack(task_mask_list, dim=0)

    # --- クルー側のパディング ---
    if C_max == 0:
        local_crew_ids = torch.empty((B, 0), dtype=torch.long, device=device)
        crew_location_station = torch.empty((B, 0), dtype=torch.long, device=device)
        crew_on_duty = torch.empty((B, 0), dtype=torch.bool, device=device)
        crew_finished = torch.empty((B, 0), dtype=torch.bool, device=device)
        crew_ready_time = torch.empty((B, 0), dtype=torch.long, device=device)
        crew_consec_work_min = torch.empty((B, 0), dtype=torch.long, device=device)
        crew_rest_remaining = torch.empty((B, 0), dtype=torch.long, device=device)
        crew_duty_minutes = torch.empty((B, 0), dtype=torch.long, device=device)
        crew_mask = torch.empty((B, 0), dtype=torch.bool, device=device)

    else:
        local_crew_ids_list = []
        crew_location_station_list = []
        crew_on_duty_list = []
        crew_ready_time_list = []
        crew_consec_work_min_list = []
        crew_rest_remaining_list = []
        crew_duty_minutes_list = []
        crew_mask_list = []

        for x in dyns:
            lc = x.local_crew_ids.to(torch.long).to(device)
            crew_location_station_i = x.crew_location_station.to(torch.long).to(device)
            crew_on_duty_i = x.crew_on_duty.to(torch.bool).to(device)
            crew_ready_time_i = x.crew_ready_time.to(torch.long).to(device)
            crew_consec_work_min_i = x.crew_consec_work_min.to(torch.long).to(device)
            crew_rest_remaining_i = x.crew_rest_remaining.to(torch.long).to(device)
            crew_duty_minutes_i = x.crew_duty_minutes.to(torch.long).to(device)

            lc_pad = _pad1d(lc, C_max, PADDING_IDX)
            crew_location_station_pad = _pad1d(crew_location_station_i, C_max, PADDING_IDX)
            crew_on_duty_pad = _pad1d(crew_on_duty_i, C_max, False)
            crew_ready_time_pad = _pad1d(crew_ready_time_i, C_max, 0)
            crew_consec_work_min_pad = _pad1d(crew_consec_work_min_i, C_max, 0)
            crew_rest_remaining_pad = _pad1d(crew_rest_remaining_i, C_max, 0)
            crew_duty_minutes_pad = _pad1d(crew_duty_minutes_i, C_max, 0)

            mask = torch.zeros((C_max,), dtype=torch.bool, device=device)
            mask[: lc.numel()] = True

            local_crew_ids_list.append(lc_pad)
            crew_location_station_list.append(crew_location_station_pad)
            crew_on_duty_list.append(crew_on_duty_pad)
            crew_ready_time_list.append(crew_ready_time_pad)
            crew_consec_work_min_list.append(crew_consec_work_min_pad)
            crew_rest_remaining_list.append(crew_rest_remaining_pad)
            crew_duty_minutes_list.append(crew_duty_minutes_pad)
            crew_mask_list.append(mask)

        local_crew_ids = torch.stack(local_crew_ids_list, dim=0)
        crew_location_station = torch.stack(crew_location_station_list, dim=0)
        crew_on_duty = torch.stack(crew_on_duty_list, dim=0)
        crew_ready_time = torch.stack(crew_ready_time_list, dim=0)
        crew_consec_work_min = torch.stack(crew_consec_work_min_list, dim=0)
        crew_rest_remaining = torch.stack(crew_rest_remaining_list, dim=0)
        crew_duty_minutes = torch.stack(crew_duty_minutes_list, dim=0)
        crew_mask = torch.stack(crew_mask_list, dim=0)
        
        # ---- crew_location_station を +1シフト（実データ位置のみ） ----
        crew_location_station_shifted = crew_location_station.clone()
        crew_location_station_shifted[crew_mask] = crew_location_station_shifted[crew_mask] + 1
        crew_location_station_shifted[~crew_mask] = 0
        crew_location_station = crew_location_station_shifted.to(torch.long)

    # --- 今回ラウンドのタスクだけを抽出して再パディング（順序維持） ---
    round_len = round_bool.sum(dim=1)                    # [B]
    R_max = int(round_len.max()) if B > 0 else 0

    if R_max == 0:
        round_local_task_ids = torch.empty((B, 0), dtype=torch.long, device=device)
        round_task_mask = torch.empty((B, 0), dtype=torch.bool, device=device)
    else:
        round_local_task_ids = torch.full((B, R_max), PADDING_IDX, dtype=torch.long, device=device)
        round_task_mask = torch.zeros((B, R_max), dtype=torch.bool, device=device)
        for b in range(B):
            k = int(round_len[b])
            if k > 0:
                sel = local_task_ids[b, round_bool[b]]  # [k]
                round_local_task_ids[b, :k] = sel
                round_task_mask[b, :k] = True

            # --- done=[B] が True の要素は、pad_masks をすべて
    
    d = done.to(torch.bool).to(device)
    if T_max > 0:
        task_mask[d] = False
    if C_max > 0:
        crew_mask[d] = False

    out: Dict[str, Any] = {
        "tasks": {
            "local_task_ids": local_task_ids,   # [B, T_max], long, PAD=pad_idx
            "round_bool": round_bool,           # [B, T_max], bool, PAD=False
            "round_local_task_ids": round_local_task_ids,  # [B, R_max], long, PAD=pad_idx
            "round_task_pad": round_task_mask, # [B, R_max], bool, PAD=False
        },
        "crews": {
            "local_crew_ids": local_crew_ids,                 # [B, C_max], long, PAD=pad_idx
            "crew_location_station": crew_location_station,    # [B, C_max], long, PAD=pad_idx
            "crew_on_duty": crew_on_duty,                      # [B, C_max], bool, PAD=False
            "crew_ready_time": crew_ready_time,                # [B, C_max], long, PAD=0
            "crew_consec_work_min": crew_consec_work_min,      # [B, C_max], long, PAD=0
            "crew_rest_remaining": crew_rest_remaining,        # [B, C_max], long, PAD=0
            "crew_duty_minutes": crew_duty_minutes,            # [B, C_max], long, PAD=0
        },
        "pad_masks": {
            "tasks": task_mask,  # [B, T_max] True=実データ, False=PAD
            "crews": crew_mask,  # [B, C_max] True=実データ, False=PAD
        },
    }

    if return_aux:
        aux = {
            "task_len": torch.tensor(T_list, dtype=torch.long, device=device),
            "crew_len": torch.tensor(C_list, dtype=torch.long, device=device),
            "T_max": torch.tensor([T_max], dtype=torch.long, device=device),
            "C_max": torch.tensor([C_max], dtype=torch.long, device=device),
        }
        out["aux"] = aux

    return out

def collate_action_mask(
    masks: List[ActionMask],
    device: Optional[torch.device] = None,
    done : Optional[torch.Tensor] = None,
    pad_task_id: int = -1,
    pad_crew_id: int = -1,
) -> Dict[str, torch.Tensor]:
    """
    入力: 各サンプルの ActionMask（matrix: [C_i, T_i]）
    出力:
      - action_mask: [B, C_max, T_max]（bool）
      - rows_crew_ids: [B, C_max]（long, PAD=pad_crew_id）
      - cols_task_ids: [B, T_max]（long, PAD=pad_task_id）
      - aux: task_len[B], crew_len[B], T_max, C_max
    """
    B = len(masks)
    if B == 0:
        if device is None:
            device = torch.device("cpu")
        return {
            "action_mask": torch.empty(0, 0, 0, dtype=torch.bool, device=device),
            "rows_crew_ids": torch.empty(0, 0, dtype=torch.long, device=device),
            "cols_task_ids": torch.empty(0, 0, dtype=torch.long, device=device),
            "aux": {
                "task_len": torch.empty(0, dtype=torch.long, device=device),
                "crew_len": torch.empty(0, dtype=torch.long, device=device),
                "T_max": torch.tensor([0], dtype=torch.long, device=device),
                "C_max": torch.tensor([0], dtype=torch.long, device=device),
            },
        }

    if device is None:
        device = masks[0].matrix.device

    T_list = [int(m.cols_task_ids.numel()) for m in masks]
    C_list = [int(m.rows_crew_ids.numel()) for m in masks]
    T_max = max(T_list)
    C_max = max(C_list)

    cols_task_ids = []
    rows_crew_ids = []
    # ここで [B, C_max, T_max] を確保（最終フォーマットに合わせる）
    mask_b = torch.zeros((B, C_max, T_max), dtype=torch.bool, device=device)

    for b, m in enumerate(masks):
        Ti = T_list[b]
        Ci = C_list[b]

        # ids をパディング
        cols = m.cols_task_ids.to(torch.long).to(device)
        rows = m.rows_crew_ids.to(torch.long).to(device)
        cols_pad = _pad1d(cols, T_max, pad_task_id)
        rows_pad = _pad1d(rows, C_max, pad_crew_id)
        cols_task_ids.append(cols_pad)
        rows_crew_ids.append(rows_pad)

        # matrix は [C_i, T_i] 前提
        mat = m.matrix.to(torch.bool).to(device)
        # 必要に応じて切り詰め（過長の場合）
        mat = mat[:Ci, :Ti]
        # ゼロ埋めして [C_max, T_max] へ
        buf = torch.zeros((C_max, T_max), dtype=torch.bool, device=device)
        buf[:Ci, :Ti] = mat
        mask_b[b] = buf
    
    # --- done=[B] が True の要素は、mask をすべて
    d = done.to(torch.bool).to(device)
    if T_max > 0 and C_max > 0:
        mask_b[d, :, :] =   False       

    out = {
        "action_mask": mask_b,  # [B, C_max, T_max]
        "rows_crew_ids": torch.stack(rows_crew_ids, dim=0) if B > 0 else torch.empty(0, C_max, dtype=torch.long, device=device),
        "cols_task_ids": torch.stack(cols_task_ids, dim=0) if B > 0 else torch.empty(0, T_max, dtype=torch.long, device=device),
        "aux": {
            "task_len": torch.tensor(T_list, dtype=torch.long, device=device),
            "crew_len": torch.tensor(C_list, dtype=torch.long, device=device),
            "T_max": torch.tensor([T_max], dtype=torch.long, device=device),
            "C_max": torch.tensor([C_max], dtype=torch.long, device=device),
        },
    }
    return out

def collate_pair_bais(pairs: List[RoundPairBias], 
                      device: Optional[torch.device],
                    done : Optional[torch.Tensor] = None,
                      ) -> Dict[str, torch.Tensor]:
    """
    RoundPairBias のリストをバッチ化して、[B, C_max, T_max] 形状にパディングする。
    返り値は PointerAttention のバイアス計算に使いやすい各チャネル辞書。
    - is_hitch, is_continuous, on_duty は bool
    - hitch_minutes, rel_time は long
    - rows_crew_ids, cols_task_ids はパディングを -1 とする
    """
    B = len(pairs)
    C_max = max((p.rows_crew_ids.numel() for p in pairs), default=0)
    T_max = max((p.cols_task_ids.numel() for p in pairs), default=0)

    if device is None:
        device = pairs[0].rows_crew_ids.device if B > 0 else torch.device("cpu")

    is_hitch = torch.zeros((B, C_max, T_max), dtype=torch.bool, device=device)
    hitch_minutes = torch.zeros((B, C_max, T_max), dtype=torch.long, device=device)
    is_continuous = torch.zeros((B, C_max, T_max), dtype=torch.bool, device=device)
    rel_time = torch.zeros((B, C_max, T_max), dtype=torch.long, device=device)
    on_duty = torch.zeros((B, C_max, T_max), dtype=torch.bool, device=device)

    rows_crew_ids = torch.full((B, C_max), -1, dtype=torch.long, device=device)
    cols_task_ids = torch.full((B, T_max), -1, dtype=torch.long, device=device)

    for b, p in enumerate(pairs):
        C = int(p.rows_crew_ids.numel())
        T = int(p.cols_task_ids.numel())

        if C > 0:
            rows_crew_ids[b, :C] = p.rows_crew_ids.to(device)
        if T > 0:
            cols_task_ids[b, :T] = p.cols_task_ids.to(device)
        if C == 0 or T == 0:
            continue

        is_hitch[b, :C, :T] = p.is_hitch.to(device)
        hitch_minutes[b, :C, :T] = p.hitch_minutes.to(device)
        is_continuous[b, :C, :T] = p.is_continuous.to(device)
        rel_time[b, :C, :T] = p.rel_time.to(device)
        on_duty[b, :C, :T] = p.on_duty.to(device)

    return {
        "rows_crew_ids": rows_crew_ids,   # [B, C_max] long, pad=-1
        "cols_task_ids": cols_task_ids,   # [B, T_max] long, pad=-1
        "is_hitch": is_hitch,             # [B, C_max, T_max] bool
        "hitch_minutes": hitch_minutes,   # [B, C_max, T_max] long
        "is_continuous": is_continuous,   # [B, C_max, T_max] bool
        "rel_time": rel_time,             # [B, C_max, T_max] long
        "on_duty": on_duty,               # [B, C_max, T_max] bool
    }
