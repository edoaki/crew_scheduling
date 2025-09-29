import argparse
from pathlib import Path
import torch
from rl_core.build import load_configs, build_generator, build_env, build_embeddings, build_policy
from rl_core.train_loop import evaluate_mean_reward

# ====== 既定値（train.py と整合） ======
DEFAULT_CONFIG_DIR = Path("test2_config")
DEFAULT_SAVE_ROOT = Path("checkpoints")
DEFAULT_BATCH_SIZE = 1

def load_policy_weights(policy: torch.nn.Module, path: Path, device: torch.device):
    obj = torch.load(str(path), map_location=device)
    # 2パターンのみ対応：
    #  (A) 直接 state_dict
    #  (B) チェックポイント dict で "policy_state_dict" を持つ
    if isinstance(obj, dict) and "policy_state_dict" in obj:
        policy.load_state_dict(obj["policy_state_dict"])
    else:
        policy.load_state_dict(obj)
    return policy

def eval(config_dir, save_root, weights, ckpt, batch_size, tag):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 構築 ===
    cfg = load_configs(config_dir)
    paths = cfg["paths"]
    constraints = cfg["constraints"]

    generator = build_generator(paths)
    station_emb, time_emb = build_embeddings(paths["station_yaml"], paths["encoding_yaml"])
    vec_env = build_env(generator, constraints, batch_size=batch_size, device=device)

    policy = build_policy(station_emb, time_emb).to(device)

    # === 重みの読み込み ===
    run_name =config_dir.name
    if ckpt is not None:
        load_policy_weights(policy, ckpt, device)
        src_path = ckpt
    elif weights is not None:
        load_policy_weights(policy, weights, device)
        src_path = weights
    else:
        # 既定は最終モデル
        final_path = save_root / run_name / f"{run_name}.final.pth"
        if not final_path.exists():
            raise FileNotFoundError(f"既定の最終モデルが見つかりません: {final_path}")
        load_policy_weights(policy, final_path, device)
        src_path = final_path

    policy.eval()
    with torch.no_grad():

        mean_reward, sampling_mean,sol,_ = evaluate_mean_reward(
            policy, vec_env, batch_size, device, mode = tag, return_env_reward=False
        )

    print("===== Evaluation Result =====")
    print(f"Model source : {src_path}")
    print(f"RUN_NAME     : {run_name}")
    print(f"Batch size   : {batch_size}")
    print(f"Mean reward  : {float(mean_reward):.6f}")
    if sampling_mean is not None:
        print(f"Sampling mean: {float(sampling_mean):.6f}")

    return sol

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

    solution = eval(
        config_dir=args.config_dir,
        save_root=args.save_root,
        weights=args.weights,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        tag=args.tag,
    )

    print("Evaluation completed.")
    print(solution)