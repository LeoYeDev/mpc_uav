"""
Simplified Ablation Experiment for Variance-Aware GP-MPC.

This script runs ablations by leveraging the existing comparative_experiment infrastructure
which has correct online GP training and simulation loops.

Run: python src/experiments/ablation_experiment.py --speed 2.7
"""
import numpy as np
import multiprocessing
from dataclasses import replace

from config.configuration_parameters import SimpleSimConfig
from config.gp_config import OnlineGPConfig
from config.paths import DEFAULT_MODEL_VERSION, DEFAULT_MODEL_NAME

# Import from comparative_experiment
from src.experiments.comparative_experiment import prepare_quadrotor_mpc, main as run_tracking
from src.gp.online import IncrementalGPManager


# ============================================================================
# Ablation Configurations
# ============================================================================

def get_ablation_configs():
    """Generate ablation configurations matching reviewer requirements."""
    base_gp_config = OnlineGPConfig()
    
    return {
        # Baseline: Nominal MPC (no GP at all)
        "nominal": {
            "description": "Nominal MPC (无GP)",
            "use_gp": False,
            "use_online_gp": False,
            "gp_config": None,
            "solver_options": {},
        },
        
        # (b) Without Online GP - Static GP only
        "static_only": {
            "description": "(b) 仅离线GP",
            "use_gp": True,
            "use_online_gp": False,
            "gp_config": None,
            "solver_options": {},
        },
        
        # (c) Without IVS - Use FIFO buffer
        "fifo_buffer": {
            "description": "(c) FIFO缓冲区",
            "use_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, buffer_type='fifo', variance_scaling_alpha=1.0),
            "solver_options": {"variance_scaling_alpha": 1.0},
        },
        
        # (d) Without Variance mechanism (alpha=0)
        "no_variance": {
            "description": "(d) 无方差机制 (α=0)",
            "use_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=0.0),
            "solver_options": {"variance_scaling_alpha": 0.0},
        },
        
        # Full System (all features enabled, α=1)
        "full_system": {
            "description": "完整系统 (α=1)",
            "use_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=1.0),
            "solver_options": {"variance_scaling_alpha": 1.0},
        },
        
        # Sensitivity: alpha = 0.5
        "alpha_05": {
            "description": "α=0.5 (轻度保守)",
            "use_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=0.5),
            "solver_options": {"variance_scaling_alpha": 0.5},
        },
        
        # Sensitivity: alpha = 2.0
        "alpha_20": {
            "description": "α=2.0 (高度保守)",
            "use_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=2.0),
            "solver_options": {"variance_scaling_alpha": 2.0},
        },
    }


def run_single_ablation(config_name, config, speed=2.7, trajectory_type="random"):
    """
    Run single ablation using the proven prepare_quadrotor_mpc and main functions.
    """
    simulation_options = SimpleSimConfig.simulation_disturbances
    
    # 根据配置决定是否使用GP
    version = DEFAULT_MODEL_VERSION if config["use_gp"] else None
    name = DEFAULT_MODEL_NAME if config["use_gp"] else None
    
    # 使用唯一的quad_name避免缓存冲突
    quad_name = f"my_quad_abl_{config_name}"
    
    # 准备MPC
    quad_mpc = prepare_quadrotor_mpc(
        simulation_options,
        version=version,
        name=name,
        reg_type="gp" if config["use_gp"] else None,
        quad_name=quad_name,
        use_online_gp=config["use_online_gp"]
    )
    
    # 如果启用在线GP，初始化管理器
    online_gp_manager = None
    if config["use_online_gp"] and config["gp_config"] is not None:
        online_gp_manager = IncrementalGPManager(config=config["gp_config"].to_dict())
    
    # 运行跟踪实验
    try:
        result = run_tracking(
            quad_mpc=quad_mpc,
            av_speed=speed,
            reference_type=trajectory_type,
            plot=False,
            use_online_gp_ject=config["use_online_gp"],
            use_wind=True,  # 使用风模型
            use_gp_ject=config["use_gp"],
            online_gp_manager=online_gp_manager
        )
        
        # 解析返回值
        rmse, max_vel, mean_opt_time, _, _, _ = result
        
        # 清理
        if online_gp_manager:
            online_gp_manager.shutdown()
        
        return {"rmse": rmse, "max_vel": max_vel, "opt_time": mean_opt_time}
    
    except Exception as e:
        print(f"  Error in {config_name}: {e}")
        if online_gp_manager:
            online_gp_manager.shutdown()
        return None


def run_ablation_study(speed=2.7, trajectory_type="random", n_seeds=1):
    """Run complete ablation study."""
    print("=" * 70)
    print("Ablation Study for Variance-Aware GP-MPC")
    print("=" * 70)
    print(f"Speed: {speed} m/s, Trajectory: {trajectory_type}, Seeds: {n_seeds}\n")

    configs = get_ablation_configs()
    results = {}

    for config_name, config in configs.items():
        print(f"Running: {config['description']}...")
        
        result = run_single_ablation(
            config_name, config, speed=speed, trajectory_type=trajectory_type
        )
        
        if result:
            results[config_name] = {
                "rmse": result["rmse"],
                "max_vel": result["max_vel"],
                "description": config['description']
            }
            print(f"  RMSE: {result['rmse']:.4f} m, Max Vel: {result['max_vel']:.2f} m/s")
        else:
            print(f"  Failed to run")

    # 打印汇总表格
    if results:
        print("\n" + "=" * 70)
        print("Summary Table")
        print("=" * 70)
        print(f"{'Configuration':<25} {'RMSE (m)':<15} {'Max Vel (m/s)':<15}")
        print("-" * 70)
        for config_name, result in results.items():
            print(f"{result['description']:<25} {result['rmse']:<15.4f} {result['max_vel']:<15.2f}")

    return results


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    import argparse
    parser = argparse.ArgumentParser(description="Ablation study for variance-aware GP-MPC")
    parser.add_argument("--speed", type=float, default=2.7, help="Average trajectory speed (m/s)")
    parser.add_argument("--trajectory", type=str, default="random", choices=["random", "loop", "lemniscate"])
    parser.add_argument("--seeds", type=int, default=1, help="Number of Monte Carlo seeds")
    args = parser.parse_args()

    results = run_ablation_study(speed=args.speed, trajectory_type=args.trajectory, n_seeds=args.seeds)
