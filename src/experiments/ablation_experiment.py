"""
Focused Ablation Experiment for FIFO vs IVS (online GP only).

This script narrows the comparison to online buffer strategies under
full GP trust (variance_scaling_alpha = 0), to isolate the effect of
the data-flow management policy itself.

Run: python src/experiments/ablation_experiment.py --speed 2.7
"""
import numpy as np
import multiprocessing
import os
import random
import matplotlib.pyplot as plt
from dataclasses import replace

from config.configuration_parameters import SimpleSimConfig
from config.gp_config import OnlineGPConfig
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
    3) IVS with novelty_weight=0 (pure recency test)

    All use alpha=0 (full GP trust) to remove variance-scaling effects.
    """
    base_gp_config = OnlineGPConfig()
    alpha = 0.0

    if preset == "contrast":
        # Controlled stress preset: keep both methods trainable while
        # shrinking memory and amplifying IVS's strength in sparse coverage.
        common = dict(
            buffer_max_size=14,
            min_points_for_initial_train=10,  # must stay <= buffer size
            refit_hyperparams_interval=6,
            worker_train_iters=24,
            recency_decay_rate=0.12,
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
            novelty_weight=0.45,
            recency_weight=0.55,
            buffer_min_distance=0.02,
            buffer_merge_min_distance=0.025,
            ivs_multilevel=True,
            buffer_level_capacities=[8, 4, 2],
            buffer_level_sparsity=[1, 3, 6],
            **common,
        )
        ivs_single_gp = replace(
            ivs_gp,
            ivs_multilevel=False,
            buffer_min_distance=0.015,
            buffer_merge_min_distance=0.015,
        )
        ivs_novelty0_gp = replace(
            ivs_gp,
            novelty_weight=0.0,
            recency_weight=1.0,
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
        ivs_single_gp = replace(
            base_gp_config,
            buffer_type='ivs',
            ivs_multilevel=False,
            variance_scaling_alpha=alpha,
        )
        ivs_novelty0_gp = replace(
            base_gp_config,
            buffer_type='ivs',
            novelty_weight=0.0,
            recency_weight=1.0,
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
        "ar_mpc_ivs_novelty0": {
            "description": "AR-MPC (IVS novelty=0, alpha=0)",
            "use_offline_gp": True,
            "use_online_gp": True,
            "gp_config": ivs_novelty0_gp,
            "solver_options": {"variance_scaling_alpha": alpha},
        },
        "ar_mpc_ivs_singlelevel": {
            "description": "AR-MPC (IVS single-level, alpha=0)",
            "use_offline_gp": True,
            "use_online_gp": True,
            "gp_config": ivs_single_gp,
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


def run_single_ablation(config_name, config, speed=2.7, trajectory_type="random",
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
    
    # Use unique quad_name to avoid ACADOS solver cache conflicts
    quad_name = f"my_quad_abl_{config_name}"
    
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
        
        # Parse results
        rmse, max_vel, mean_opt_time, _, _, _ = result
        
        # Cleanup
        if online_gp_manager:
            online_gp_manager.shutdown()
        
        return {"rmse": rmse, "max_vel": max_vel, "opt_time": mean_opt_time}
    
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
        except:
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
    configs = list(results.keys())
    # Defined colors
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E']
    
    # ========== Figure 1: RMSE Bar Chart (Full) ==========
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
    
    # ========== Figure 1b: Zoomed RMSE Bar Chart (Excluding Nominal & SGP) ==========
    # Filter for AR-MPC variants to highlight IVS vs FIFO and Alpha sensitivity
    zoom_keys = [k for k in configs if 'ar_mpc' in k or 'sensitivity' in k]
    if len(zoom_keys) > 1:
        zoom_names = [results[k]['description'] for k in zoom_keys]
        zoom_rmses = [results[k]['rmse'] for k in zoom_keys]
        
        # Use a subset of colors
        zoom_colors = [colors[configs.index(k) % len(colors)] for k in zoom_keys]
        
        fig1b, ax1b = plt.subplots(figsize=(8, 5))
        bars1b = ax1b.bar(range(len(zoom_names)), zoom_rmses, color=zoom_colors, edgecolor='black', linewidth=1.2)
        
        ax1b.set_xticks(range(len(zoom_names)))
        ax1b.set_xticklabels(zoom_names, rotation=30, ha='right', fontsize=10)
        ax1b.set_ylabel('RMSE (m)', fontsize=12)
        ax1b.set_title(f'Ablation Detail: AR-MPC Variants (Zoomed)', fontsize=14)
        
        # Smart Y-axis limits to highlight differences
        min_r = min(zoom_rmses)
        max_r = max(zoom_rmses)
        margin = (max_r - min_r) * 0.5 if max_r > min_r else 0.005
        ax1b.set_ylim(max(0, min_r - margin), max_r + margin)

        # Add value labels
        for bar, rmse in zip(bars1b, zoom_rmses):
             ax1b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + margin*0.05, 
                f'{rmse:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1b.grid(axis='y', alpha=0.3, which='both')
        plt.tight_layout()
        
        fig1b_path = os.path.join(save_dir, 'ablation_rmse_zoomed.pdf')
        fig1b.savefig(fig1b_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig1b_path}")
        plt.close(fig1b)

    
    # ========== Figure 2: Relative Improvement Chart ==========
    if 'nominal' in results:
        baseline = results['nominal']['rmse']
        improvements = [(1 - r['rmse']/baseline) * 100 for r in results.values()]
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars2 = ax2.bar(range(len(names)), improvements, color=colors[:len(names)], 
                        edgecolor='black', linewidth=1.2)
        
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
        ax2.set_ylabel('Improvement over Nominal (%)', fontsize=12)
        ax2.set_title(f'Relative Improvement by Configuration', fontsize=14)
        
        # Add value labels
        for bar, imp in zip(bars2, improvements):
            # Place label above bar for positive, below for negative
            y_pos = bar.get_height() + 1 if imp >= 0 else bar.get_height() - 5
            va = 'bottom' if imp >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    f'{imp:.1f}%', ha='center', va=va, fontsize=9)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        fig2_path = os.path.join(save_dir, 'ablation_improvement.pdf')
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig2_path}")
        plt.close(fig2)
    
    # ========== Figure 3: Alpha Sensitivity Analysis ==========
    # Filter keys containing 'alpha' or the boundary cases
    alpha_keys = [k for k in results.keys() if 'alpha' in k or k in ['ar_mpc_no_var', 'ar_mpc']]
    
    if len(alpha_keys) > 1:
        # Extract alpha values and corresponding RMSEs
        alpha_data = []
        for name in alpha_keys:
            rmse = results[name]['rmse']
            if name == 'ar_mpc_no_var':
                alpha_data.append((0.0, rmse))
            elif name == 'ar_mpc':
                alpha_data.append((1.0, rmse))
            elif 'sensitivity_alpha_05' in name:
                alpha_data.append((0.5, rmse))
            elif 'sensitivity_alpha_20' in name:
                alpha_data.append((2.0, rmse))
        
        if alpha_data:
            alpha_data.sort(key=lambda x: x[0])
            alphas, rmse_vals = zip(*alpha_data)
            
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.plot(alphas, rmse_vals, 'o-', markersize=10, linewidth=2, color='#2E86AB')
            
            # Highlight best point
            min_idx = rmse_vals.index(min(rmse_vals))
            ax3.scatter([alphas[min_idx]], [min(rmse_vals)], 
                       s=150, c='#E74C3C', marker='*', zorder=5, label='Best')
            
            ax3.set_xlabel('Variance Scaling Alpha', fontsize=12)
            ax3.set_ylabel('RMSE (m)', fontsize=12)
            ax3.set_title('Sensitivity Analysis: Variance Scaling Parameter', fontsize=14)
            # Add zoomed Y-axis to see small differences
            min_r = min(rmse_vals)
            max_r = max(rmse_vals)
            margin = (max_r - min_r) * 0.5 if max_r > min_r else 0.005
            ax3.set_ylim(max(0, min_r - margin), max_r + margin)

            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            plt.tight_layout()
            fig3_path = os.path.join(save_dir, 'alpha_sensitivity.pdf')
            fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {fig3_path}")
            plt.close(fig3)


def run_ablation_study(speed=2.7, trajectory_type="random", n_seeds=1, visualize=True,
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
        "ar_mpc_ivs_singlelevel",
        "ar_mpc_ivs",
        "ar_mpc_ivs_novelty0",
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
            rmse_mean = float(np.mean(rmse_values))
            rmse_std = float(np.std(rmse_values))

            results[config_name] = {
                "rmse": rmse_mean,
                "rmse_std": rmse_std,
                "max_vel": float(np.mean(max_vel_values)),
                "description": config['description']
            }
            print(
                f"  RMSE(mean±std): {rmse_mean:.4f} ± {rmse_std:.4f} m, "
                f"Max Vel(mean): {np.mean(max_vel_values):.2f} m/s"
            )
        else:
            print(f"  Failed to run {config_name}")

    # Print Summary Table
    if results:
        print("\n" + "=" * 70)
        print("Summary Table")
        print("=" * 70)
        print(f"{'Configuration':<35} {'RMSE mean±std (m)':<24} {'Max Vel (m/s)':<15}")
        print("-" * 70)
        for config_name in ordered_keys:
            if config_name in results:
                result = results[config_name]
                print(
                    f"{result['description']:<35} "
                    f"{result['rmse']:.4f}±{result.get('rmse_std', 0.0):.4f}{'':<8} "
                    f"{result['max_vel']:<15.2f}"
                )
        
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
    parser.add_argument("--speed", type=float, default=3.5, help="Average trajectory speed (m/s)")
    parser.add_argument("--trajectory", type=str, default="random", choices=["random", "loop", "lemniscate"])
    parser.add_argument("--seeds", type=int, default=1, help="Number of Monte Carlo seeds")
    parser.add_argument("--seed-base", type=int, default=303, help="Base random seed")
    parser.add_argument("--wind-profile", type=str, default="default", choices=["default", "regime_shift"])
    parser.add_argument("--preset", type=str, default="contrast", choices=["baseline", "contrast"])
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
