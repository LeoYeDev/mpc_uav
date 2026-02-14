"""
Focused Ablation Experiment for FIFO vs IVS (online GP only).

This script narrows the comparison to online buffer strategies under
full GP trust (variance_scaling_alpha = 0), to isolate the effect of
the data-flow management policy itself.

Run: python src/experiments/ablation_experiment.py --speed 3.0
"""
import numpy as np
import multiprocessing
import os
import random
import re
import matplotlib.pyplot as plt
from dataclasses import replace

from config.configuration_parameters import SimpleSimConfig
from config.gp_config import build_online_gp_config
from config.paths import DEFAULT_MODEL_VERSION, DEFAULT_MODEL_NAME

import warnings
try:
    from gpytorch.utils.warnings import NumericalWarning
    warnings.simplefilter("ignore", NumericalWarning)
except ImportError:
    pass
try:
    import torch
except ImportError:
    torch = None

# Import from comparative_experiment
from src.experiments.comparative_experiment import prepare_quadrotor_mpc, main as run_tracking
from src.gp.online import IncrementalGPManager


# ============================================================================
# Ablation Configurations
# ============================================================================

def get_ablation_configs(preset="baseline"):
    """
    Focused configs:
    1) FIFO
    2) IVS (default weights)

    All use alpha=0 (full GP trust) to isolate data management effects.
    """
    alpha = 0.0
    # 统一从集中配置派生，确保与 online gp/buffer 模块保持同步。
    base_gp_config = build_online_gp_config(
        buffer_type='ivs',
        async_hp_updates=True,
        variance_scaling_alpha=alpha,
    )

    if preset == "contrast":
        # Controlled stress preset: keep both methods trainable while
        # shrinking memory and amplifying IVS's sparse-coverage advantage.
        common = dict(
            buffer_max_size=12,
            min_points_for_initial_train=8,  # must stay <= buffer size
            refit_hyperparams_interval=12,
            worker_train_iters=12,
        )
        fifo_gp = replace(
            base_gp_config,
            buffer_type='fifo',
            variance_scaling_alpha=alpha,
            **common,
        )
        ivs_gp = replace(
            base_gp_config,
            buffer_type='ivs',
            variance_scaling_alpha=alpha,
            buffer_insert_min_delta_v=0.15,
            buffer_prune_old_delta_v=0.15,
            buffer_flip_prune_limit=3,
            **common,
        )
    else:
        fifo_gp = replace(
            base_gp_config,
            buffer_type='fifo',
            variance_scaling_alpha=alpha,
        )
        ivs_gp = replace(
            base_gp_config,
            buffer_type='ivs',
            variance_scaling_alpha=alpha,
        )

    return {
        "ar_mpc_fifo": {
            "description": "AR-MPC (FIFO, alpha=0)",
            "use_offline_gp": True,
            "use_online_gp": True,
            "gp_config": fifo_gp,
            "solver_options": {"variance_scaling_alpha": alpha},
        },
        "ar_mpc_ivs": {
            "description": "AR-MPC (IVS, alpha=0)",
            "use_offline_gp": True,
            "use_online_gp": True,
            "gp_config": ivs_gp,
            "solver_options": {"variance_scaling_alpha": alpha},
        },
    }


def _set_global_seed(seed: int, seed_cuda: bool = False) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if seed_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def run_single_ablation(config_name, config, speed=3.0, trajectory_type="random",
                        seed=303, wind_profile="default"):
    """
    Run single ablation using the proven prepare_quadrotor_mpc and main functions.
    """
    # Keep CPU-side determinism by default; avoid touching CUDA RNG unless needed.
    _set_global_seed(int(seed), seed_cuda=False)
    simulation_options = SimpleSimConfig.simulation_disturbances
    
    # Configure Offline Model Loading
    # If use_offline_gp is True, we load the pre-trained SGP model.
    version = DEFAULT_MODEL_VERSION if config["use_offline_gp"] else None
    name = DEFAULT_MODEL_NAME if config["use_offline_gp"] else None
    reg_type = "gp" if config["use_offline_gp"] else None
    
    # Use unique quad_name to avoid ACADOS solver cache conflicts.
    # CasADi function names only allow [A-Za-z0-9_] and cannot start with digits.
    safe_name = re.sub(r'[^A-Za-z0-9_]+', '_', str(config_name))
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')
    if not safe_name or safe_name[0].isdigit():
        safe_name = f"cfg_{safe_name}"
    quad_name = f"my_quad_abl_{safe_name}"
    
    # Prepare MPC Controller
    quad_mpc = prepare_quadrotor_mpc(
        simulation_options,
        version=version,
        name=name,
        reg_type=reg_type,
        quad_name=quad_name,
        use_online_gp=config["use_online_gp"],
        solver_options=config.get("solver_options", None)
    )
    
    # Prepare Online GP Manager if enabled
    online_gp_manager = None
    if config["use_online_gp"] and config["gp_config"] is not None:
        online_gp_manager = IncrementalGPManager(config=config["gp_config"].to_dict())
    
    # Run Tracking Experiment
    try:
        # Note: use_gp_ject in main() corresponds to whether we inject the *offline* GP
        # use_online_gp_ject corresponds to whether we inject the *online* GP
        result = run_tracking(
            quad_mpc=quad_mpc,
            av_speed=speed,
            reference_type=trajectory_type,
            plot=False,
            use_online_gp=config["use_online_gp"],
            use_offline_gp=config["use_offline_gp"], 
            online_gp_manager=online_gp_manager,
            model_label=config.get("description", config_name),
            wind_profile=wind_profile,
            trajectory_seed=int(seed),
        )
        
        # Parse results (backward-compatible with extended latency outputs).
        rmse, max_vel, mean_opt_time, _, _, _, *extras = result
        gp_update_time = float(extras[0]) if len(extras) >= 1 else 0.0
        control_time = float(extras[1]) if len(extras) >= 2 else float(mean_opt_time)
        runtime_details = extras[2] if len(extras) >= 3 and isinstance(extras[2], dict) else {}
        gp_predict_time = float(runtime_details.get("gp_predict_time", np.nan))
        buffer_update_time = float(runtime_details.get("buffer_update_time", np.nan))
        queue_overhead_time = float(runtime_details.get("queue_overhead_time", np.nan))

        # Cleanup
        if online_gp_manager:
            online_gp_manager.shutdown()
        
        return {
            "rmse": rmse,
            "max_vel": max_vel,
            "opt_time": mean_opt_time,            # Solve-only latency
            "gp_update_time": gp_update_time,     # Online update/poll overhead
            "control_time": control_time,         # Solve + online overhead
            "gp_predict_time": gp_predict_time,   # Online prediction overhead
            "buffer_update_time": buffer_update_time,
            "queue_overhead_time": queue_overhead_time,
        }
    
    except Exception as e:
        print(f"  Error in {config_name}: {e}")
        import traceback
        traceback.print_exc()
        if online_gp_manager:
            online_gp_manager.shutdown()
        return None


def plot_ablation_results(results, speed, save_dir="outputs/figures"):
    """
    Generate and save ablation result visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Force standard academic font to avoid "SimHei" errors
    import matplotlib
    matplotlib.rc('font', family='serif') 
    
    # Try to use seaborn style if available for better aesthetics
    # Fix for MatplotlibDeprecationWarning
    plt.style.use('default')
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid']:
        try:
            plt.style.use(style)
            break
        except Exception:
            pass
    
    # Re-enforce font family after style change
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Liberation Serif']

    if not results:
        print("No results to plot")
        return
    
    # Prepare data
    names = [r['description'] for r in results.values()]
    rmses = [r['rmse'] for r in results.values()]
    ctrl_ms = [float(r.get('control_time', np.nan)) * 1000.0 for r in results.values()]
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

    # ========== Figure 1: RMSE Bar Chart ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(range(len(names)), rmses, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
    
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax1.set_ylabel('RMSE (m)', fontsize=12)
    ax1.set_title(f'Ablation Study Results (Speed: {speed} m/s)', fontsize=14)
    if rmses:
        ax1.set_ylim(0, max(rmses) * 1.2)
    
    # Add value labels
    for bar, rmse in zip(bars, rmses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{rmse:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    fig1_path = os.path.join(save_dir, 'ablation_rmse_comparison.pdf')
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig1_path}")
    plt.close(fig1)

    # ========== Figure 2: Control Time Bar Chart ==========
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars2 = ax2.bar(range(len(names)), ctrl_ms, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax2.set_ylabel('Control Compute Time (ms)', fontsize=12)
    ax2.set_title(f'Latency Comparison (Speed: {speed} m/s)', fontsize=14)
    if np.isfinite(np.nanmax(ctrl_ms)):
        ax2.set_ylim(0.0, float(np.nanmax(ctrl_ms)) * 1.2 + 1e-6)
    for bar, lat in zip(bars2, ctrl_ms):
        if np.isfinite(lat):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{lat:.2f}', ha='center', va='bottom', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    fig2_path = os.path.join(save_dir, 'ablation_control_time_comparison.pdf')
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig2_path}")
    plt.close(fig2)


def run_ablation_study(speed=3.0, trajectory_type="random", n_seeds=1, visualize=True,
                       wind_profile="default", seed_base=303, preset="baseline"):
    """Run complete ablation study."""
    print("=" * 70)
    print("Focused Ablation: FIFO vs IVS (alpha=0)")
    print("=" * 70)
    print(
        f"Speed: {speed} m/s, Trajectory: {trajectory_type}, "
        f"Seeds: {n_seeds}, Wind: {wind_profile}, Preset: {preset}\n"
    )

    configs = get_ablation_configs(preset=preset)
    results = {}

    # Define order of execution/plotting
    ordered_keys = [
        "ar_mpc_fifo",
        "ar_mpc_ivs",
    ]
    
    # Ensure all keys exist
    ordered_keys = [k for k in ordered_keys if k in configs]

    for config_name in ordered_keys:
        config = configs[config_name]
        print(f"Running: {config['description']}...")

        seed_results = []
        for seed_offset in range(max(1, int(n_seeds))):
            seed = int(seed_base) + seed_offset
            result = run_single_ablation(
                config_name,
                config,
                speed=speed,
                trajectory_type=trajectory_type,
                seed=seed,
                wind_profile=wind_profile,
            )
            if result is not None:
                seed_results.append(result)

        if seed_results:
            rmse_values = [r["rmse"] for r in seed_results]
            max_vel_values = [r["max_vel"] for r in seed_results]
            ctrl_values = [r.get("control_time", np.nan) for r in seed_results]
            solve_values = [r.get("opt_time", np.nan) for r in seed_results]
            update_values = [r.get("gp_update_time", np.nan) for r in seed_results]
            rmse_mean = float(np.mean(rmse_values))
            rmse_std = float(np.std(rmse_values))

            results[config_name] = {
                "rmse": rmse_mean,
                "rmse_std": rmse_std,
                "max_vel": float(np.mean(max_vel_values)),
                "control_time": float(np.nanmean(ctrl_values)),
                "solve_time": float(np.nanmean(solve_values)),
                "update_time": float(np.nanmean(update_values)),
                "description": config['description']
            }
            print(
                f"  RMSE(mean±std): {rmse_mean:.4f} ± {rmse_std:.4f} m, "
                f"Max Vel(mean): {np.mean(max_vel_values):.2f} m/s, "
                f"Control(mean): {np.nanmean(ctrl_values)*1000.0:.2f} ms"
            )
        else:
            print(f"  Failed to run {config_name}")

    # Print Summary Table
    if results:
        print("\n" + "=" * 70)
        print("Summary Table")
        print("=" * 70)
        print(f"{'Configuration':<35} {'RMSE mean±std (m)':<24} {'Ctrl (ms)':<10} {'Max Vel (m/s)':<15}")
        print("-" * 70)
        for config_name in ordered_keys:
            if config_name in results:
                result = results[config_name]
                print(
                    f"{result['description']:<35} "
                    f"{result['rmse']:.4f}±{result.get('rmse_std', 0.0):.4f}{'':<8} "
                    f"{result.get('control_time', np.nan)*1000.0:<10.2f} "
                    f"{result['max_vel']:<15.2f}"
                )

        if "ar_mpc_fifo" in results and "ar_mpc_ivs" in results:
            fifo_rmse = float(results["ar_mpc_fifo"]["rmse"])
            ivs_rmse = float(results["ar_mpc_ivs"]["rmse"])
            improve_pct = (fifo_rmse - ivs_rmse) / max(fifo_rmse, 1e-12) * 100.0
            print("-" * 70)
            print(f"IVS vs FIFO RMSE improvement: {improve_pct:+.2f}%")

        # Visualize
        if visualize:
            print("\n--- Generating visualizations ---")
            plot_ablation_results(results, speed)

    return results


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    import argparse
    parser = argparse.ArgumentParser(description="Ablation study for FIFO vs IVS (alpha=0)")
    parser.add_argument("--speed", type=float, default=3.0, help="Average trajectory speed (m/s)")
    parser.add_argument("--trajectory", type=str, default="random", choices=["random", "loop", "lemniscate"])
    parser.add_argument("--seeds", type=int, default=1, help="Number of Monte Carlo seeds")
    parser.add_argument("--seed-base", type=int, default=303, help="Base random seed")
    parser.add_argument("--wind-profile", type=str, default="default", choices=["default", "regime_shift"])
    parser.add_argument("--preset", type=str, default="baseline", choices=["baseline", "contrast"])
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    results = run_ablation_study(
        speed=args.speed, 
        trajectory_type=args.trajectory, 
        n_seeds=args.seeds,
        visualize=not args.no_viz,
        wind_profile=args.wind_profile,
        seed_base=args.seed_base,
        preset=args.preset,
    )
