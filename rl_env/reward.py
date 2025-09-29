import numpy as np
import torch


def calculate_reward(sol,env_reward, vec_env,device, return_components=False,coeffs=None):
    """
    sol: LongTensor [B, T]  各タスクに割り当てられたクルーID。未割当は -1
    vec_env: VecCrewAREnv のインスタンス（envs[i].static を持つ）

    coeffs: dict or None
        {"work": float, "hitch": float, "unassigned": float}
        デフォルトは {"work":1.0, "hitch":1.0, "unassigned":1000.0}
        （未割当ペナルティは強めが推奨）

    device: torch device または None（None のとき sol.device に合わせる）

    return_components: True のとき、各コンポーネントのテンソルも返す

    戻り値:
        reward: Tensor [B]
        （return_components=True のときは (reward, components) を返し、
         components は dict[str, Tensor[B]]）
    """
    if coeffs is None:
        # coeffs = {"work": 0.02, "hitch": 0.1, "unassigned": 10}
        coeffs = {"work": 0, "hitch": 0, "unassigned": 1.0,"env_reward":0.1}

    B = len(sol)

    work_time = torch.zeros(B, dtype=torch.float32, device=device)
    hitch_time = torch.zeros(B, dtype=torch.float32, device=device)
    unassigned = torch.zeros(B, dtype=torch.float32, device=device)

    # evaluate_solution は個別環境ごとの評価を返す前提（dict）
    # {"total_work_time": int, "total_hitch_time": int, "unassigned_count": int}
    for i in range(B):
        env = vec_env.envs[i]
        s = env.static
        sol_i = sol[i]
        result = evaluate_solution(s, sol_i)
        
        work_time[i] = float(result["total_work_time"])
        hitch_time[i] = float(result["total_hitch_time"])
        unassigned[i] = float(result["unassigned_count"])

        # print(f"Batch {i}: work {work_time[i]}, hitch {hitch_time[i]}, unassigned {unassigned[i]}")

    # コスト = a*work + b*hitch + c*unassigned
    cost = (
        coeffs["work"] * work_time
        + coeffs["hitch"] * hitch_time
        + coeffs["unassigned"] * unassigned
    )
    env_reward = coeffs["env_reward"] * env_reward.to(device)
    reward = -cost + env_reward

    if return_components:
        comps = {
            "env_reward": env_reward,
            "cost": cost,
        }
        return reward, comps
    return reward 
    

def evaluate_solution(s, sol):
    """
    s: 環境の状態（s.num_tasks, s.num_crews, 各種[T]配列と[T,S]行列, [C]配列を持つ想定）
    sol: shape [T] 各タスクに割り当てられたクルーID。未割当は -1

    戻り値:
      result = {
        "did_work":            [C] bool
        "current_station":     [C] int  （最終的な現在地）
        "shift_start_time":    [C] int  （勤務開始時刻。未勤務は -1）
        "final_work_time":     [C] int  （最終勤務時刻。未勤務は -1）
        "hitch_time":          [C] int  （便乗時間の合計）
        "hitch_count":         [C] int  （便乗回数）
        "unassigned_count":    int      （sol に -1 が入っていた回数）
        "total_work_time":     int      （各クルーの(最終勤務-開始)の合計。未勤務は0）
        "total_hitch_time":    int      （便乗時間の総和）
      }
    """
    T = s.num_tasks
    C = s.num_crews

    did_work = np.zeros(C, dtype=bool)
    current_station = np.array(s.start_station_idx, copy=True)  # [C]
    shift_start_time = np.array(s.assignable_start_min, copy=True)  # [C]
    final_work_time = np.full(C, -1, dtype=int)
    hitch_time = np.zeros(C, dtype=int)
    hitch_count = np.zeros(C, dtype=int)

    unassigned_count = 0

    # タスクは開始時間順に並んでいる前提でそのまま処理
    for t in range(T):
        crew = sol[t]
        if crew == -1:
            unassigned_count += 1
            continue

        cs = current_station[crew]  # そのクルーの現在駅

        # 便乗判定：is_hitch[t, 現在駅] が 1 のとき便乗が必要で可行
        if s.is_hitch[t, cs] == 1:
            ht = int(s.hitch_minutes[t, cs])
            hitch_time[crew] += ht
            hitch_count[crew] += 1
            start_candidate = int(s.depart_time[t]) - ht
        else:
            start_candidate = int(s.depart_time[t])

        # まだ勤務していなければ勤務開始をセット
        if not did_work[crew]:
            did_work[crew] = True
            shift_start_time[crew] = start_candidate

        # 毎タスクで最終勤務時間と現在地を更新
        final_work_time[crew] = int(s.arrive_time[t])
        current_station[crew] = int(s.arrive_station[t])

    # 勤務時間（未勤務は 0）
    work_durations = np.where(did_work, final_work_time - shift_start_time, 0)
    total_work_time = int(work_durations.sum())
    total_hitch_time = int(hitch_time.sum())

    return {
        "did_work": did_work,
        "current_station": current_station,
        "shift_start_time": shift_start_time,
        "final_work_time": final_work_time,
        "hitch_time": hitch_time,
        "hitch_count": hitch_count,
        "unassigned_count": unassigned_count,
        "total_work_time": total_work_time,
        "total_hitch_time": total_hitch_time,
    }
