"""
Simplified Ablation Experiment for Variance-Aware GP-MPC.

This script runs ablations by leveraging the existing comparative_experiment infrastructure
which has correct online GP training and simulation loops.

Run: python src/experiments/ablation_experiment.py --speed 2.7
"""
import numpy as np
import multiprocessing
import os
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
            "description": "Nominal MPC",
            "use_offline_gp": False,
            "use_online_gp": False,
            "gp_config": None,
            "solver_options": {},
        },
        
        # (b) Without Online GP - Static GP only (SGP-MPC)
        "sgp_mpc": {
            "description": "SGP-MPC (Offline Only)",
            "use_offline_gp": True,
            "use_online_gp": False,
            "gp_config": None,
            "solver_options": {},
        },
        
        # (c) Without IVS - Use FIFO buffer (AR-MPC with FIFO)
        "ar_mpc_fifo": {
            "description": "AR-MPC (FIFO Buffer)",
            "use_offline_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, buffer_type='fifo', variance_scaling_alpha=1.0),
            "solver_options": {"variance_scaling_alpha": 1.0},
        },
        
        # (d) Without Variance mechanism (alpha=0) (AR-MPC No Var)
        "ar_mpc_no_var": {
            "description": "AR-MPC (No Variance)",
            "use_offline_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=0.0),
            "solver_options": {"variance_scaling_alpha": 0.0},
        },
        
        # Full System (all features enabled, α=1) (AR-MPC)
        "ar_mpc": {
            "description": "AR-MPC (Proposed)",
            "use_offline_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=1.0),
            "solver_options": {"variance_scaling_alpha": 1.0},
        },
        
        # Sensitivity: alpha = 0.5
        "sensitivity_alpha_05": {
            "description": "AR-MPC (Alpha=0.5)",
            "use_offline_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=0.5),
            "solver_options": {"variance_scaling_alpha": 0.5},
        },
        
        # Sensitivity: alpha = 2.0
        "sensitivity_alpha_20": {
            "description": "AR-MPC (Alpha=2.0)",
            "use_offline_gp": True,
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
            model_label=config.get("description", config_name)
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
    fig1.savefig(fig1_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
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
        fig1b.savefig(fig1b_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
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


def run_ablation_study(speed=2.7, trajectory_type="random", n_seeds=1, visualize=True):
    """Run complete ablation study."""
    print("=" * 70)
    print("Ablation Study for Variance-Aware GP-MPC")
    print("=" * 70)
    print(f"Speed: {speed} m/s, Trajectory: {trajectory_type}, Seeds: {n_seeds}\n")

    configs = get_ablation_configs()
    results = {}

    # Define order of execution/plotting
    ordered_keys = [
        "nominal", "sgp_mpc", "ar_mpc", 
        "ar_mpc_fifo", "ar_mpc_no_var", 
        "sensitivity_alpha_05", "sensitivity_alpha_20"
    ]
    
    # Ensure all keys exist
    ordered_keys = [k for k in ordered_keys if k in configs]

    for config_name in ordered_keys:
        config = configs[config_name]
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
            print(f"  Failed to run {config_name}")

    # Print Summary Table
    if results:
        print("\n" + "=" * 70)
        print("Summary Table")
        print("=" * 70)
        print(f"{'Configuration':<25} {'RMSE (m)':<15} {'Max Vel (m/s)':<15}")
        print("-" * 70)
        for config_name in ordered_keys:
            if config_name in results:
                result = results[config_name]
                print(f"{result['description']:<25} {result['rmse']:<15.4f} {result['max_vel']:<15.2f}")
        
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
    parser = argparse.ArgumentParser(description="Ablation study for variance-aware GP-MPC")
    parser.add_argument("--speed", type=float, default=2.7, help="Average trajectory speed (m/s)")
    parser.add_argument("--trajectory", type=str, default="random", choices=["random", "loop", "lemniscate"])
    parser.add_argument("--seeds", type=int, default=303, help="Number of Monte Carlo seeds")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()

    results = run_ablation_study(
        speed=args.speed, 
        trajectory_type=args.trajectory, 
        n_seeds=args.seeds,
        visualize=not args.no_viz
    )
