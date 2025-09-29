import os
from pathlib import Path
import argparse
import csv
import torch
from rl_core.build import load_configs, build_generator, build_env, build_embeddings, build_policy
from rl_core.train_loop import reinforce_step, evaluate_mean_reward, save_checkpoint, checkpoint_path
from utils.see import draw_timetable_with_crew
from utils.task_assign import draw_timetable_with_crew_assign
from models.embedding.utils import load_station_time_from_A
from eval import eval
# ====== 設定（必要に応じて run.yaml から読み込むように変えてもOK） ======
DEFAULT_CONFIG_DIR = Path("test2_config")
DEFAULT_SAVE_ROOT = Path("checkpoints")
DEFAULT_BATCH_SIZE = 1

def main(config_dir, save_root, weights, ckpt, batch_size, tag):

    cfg = load_configs(config_dir)
    paths = cfg["paths"]

    generator = build_generator(paths)

    data = generator.generate()
    
    depart_station = data["depart_station"].tolist()
    arrive_station = data["arrive_station"].tolist()
    depart_time = data["depart_time"].tolist()
    arrive_time = data["arrive_time"].tolist()
    crew_start_station_idx = data["start_station_idx"].tolist()
    crew_assignable_start_min = data["assignable_start_min"].tolist()

    config_dir = Path("test2_config")
    station_yaml = str(config_dir / "station.yaml")
    encoding_yaml = str(config_dir / "encoding.yaml")
    station_time_from_A = load_station_time_from_A(station_yaml, encoding_yaml)

    sol = eval(
        config_dir=config_dir,
        save_root=save_root,
        weights=weights,
        ckpt=ckpt,
        batch_size=batch_size,
        tag=tag,
    )
    solution = sol[0]
    print("solution",solution)
     # ====== ダイヤの可視化 ======

    # draw_timetable_with_crew(
    # depart_station=depart_station,
    # arrive_station=arrive_station,
    # depart_time=depart_time,
    # arrive_time=arrive_time,
    # crew_start_station_idx=crew_start_station_idx,
    # crew_assignable_start_min=crew_assignable_start_min,
    # station_time_from_A=station_time_from_A,
    # )
    draw_timetable_with_crew_assign(
    depart_station=depart_station,
    arrive_station=arrive_station,
    depart_time=depart_time,
    arrive_time=arrive_time,
    crew_start_station_idx=crew_start_station_idx,
    crew_assignable_start_min=crew_assignable_start_min,
    station_time_from_A=station_time_from_A,
    solution=solution,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained policy.")
    parser.add_argument("--config_dir", type=Path, default=DEFAULT_CONFIG_DIR,
                        help="設定ディレクトリ（run.yaml 等がある場所）。RUN_NAME はこのディレクトリ名。")
    parser.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT,
                        help="チェックポイント保存ルート。既定は checkpoints/")
    parser.add_argument("--weights", type=Path, default=None,
                        help="state_dict 形式の .pth ファイルへのパス（例：.../RUN_NAME.final.pth）")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="save_checkpoint で保存したチェックポイントへのパス（policy_state_dict を含む想定）")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="評価時のバッチサイズ")
    parser.add_argument("--tag", type=str, default="model",
                        help="evaluate_mean_reward に渡すタグ（ログ用）。既定は 'model'")
    args = parser.parse_args()

    main(
        config_dir=args.config_dir,
        save_root=args.save_root,
        weights=args.weights,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        tag=args.tag,
    )