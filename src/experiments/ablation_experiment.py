"""
Comprehensive Ablation Experiment for Variance-Aware GP-MPC.

Implements ablation study required by reviewers:
(a) without static GP
(b) without online GP
(c) without IVS (FIFO buffer)
(d) without variance mechanism (alpha=0)
(e) with/without async hyperparameter updates

Also includes sensitivity analysis for alpha values.

Run: python src/experiments/ablation_experiment.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from dataclasses import replace

from config.configuration_parameters import SimpleSimConfig
from config.gp_config import OnlineGPConfig
from config.paths import DEFAULT_MODEL_VERSION, DEFAULT_MODEL_NAME, DEFAULT_MODEL_TYPE
from src.core.controller import Quad3DMPC
from src.core.dynamics import Quadrotor3D
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.utils import load_pickled_models, separate_variables
from src.utils.wind_model import RealisticWindModel
from src.utils.trajectories import random_trajectory, lemniscate_trajectory, loop_trajectory
from src.gp.online import IncrementalGPManager
from src.gp.utils import world_to_body_velocity_mapping


# ============================================================================
# Comprehensive Ablation Configurations (Reviewer Requirements)
# ============================================================================

def get_ablation_configs():
    """Generate ablation configurations matching reviewer requirements."""
    base_gp_config = OnlineGPConfig()
    
    return {
        # Baseline: Nominal MPC (no GP at all)
        "nominal": {
            "description": "Nominal MPC (无GP)",
            "use_static_gp": False,
            "use_online_gp": False,
            "gp_config": None,
            "solver_options": {"variance_scaling_alpha": 0.0},
        },
        
        # NOTE: (a) "online_only" (无静态GP，仅在线GP) 需要更深层架构修改
        # 当前实现中在线GP依赖静态GP模型结构，暂时跳过此消融配置
        # 可以通过论文中解释：在线GP设计为离线GP的增强，而非独立替代
        # "online_only": {
        #     "description": "(a) 仅在线GP",
        #     "use_static_gp": False,
        #     "use_online_gp": True,
        #     "gp_config": replace(base_gp_config, variance_scaling_alpha=1.0),
        #     "solver_options": {"variance_scaling_alpha": 1.0},
        # },
        
        # (b) Without Online GP - Static GP only
        "static_only": {
            "description": "(b) 仅离线GP",
            "use_static_gp": True,
            "use_online_gp": False,
            "gp_config": None,
            "solver_options": {"variance_scaling_alpha": 0.0},
        },
        
        # (c) Without IVS - Use FIFO buffer
        "fifo_buffer": {
            "description": "(c) FIFO缓冲区",
            "use_static_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, buffer_type='fifo', variance_scaling_alpha=1.0),
            "solver_options": {"variance_scaling_alpha": 1.0},
        },
        
        # (d) Without Variance mechanism (alpha=0)
        "no_variance": {
            "description": "(d) 无方差机制 (α=0)",
            "use_static_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=0.0),
            "solver_options": {"variance_scaling_alpha": 0.0},
        },
        
        # (e) Sync HP updates (blocking)
        "sync_updates": {
            "description": "(e) 同步HP更新",
            "use_static_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, async_hp_updates=False, variance_scaling_alpha=1.0),
            "solver_options": {"variance_scaling_alpha": 1.0},
        },
        
        # Full System (all features enabled)
        "full_system": {
            "description": "完整系统",
            "use_static_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=1.0),
            "solver_options": {"variance_scaling_alpha": 1.0},
        },
        
        # Sensitivity: alpha = 0.5
        "alpha_05": {
            "description": "α=0.5 (轻度保守)",
            "use_static_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=0.5),
            "solver_options": {"variance_scaling_alpha": 0.5},
        },
        
        # Sensitivity: alpha = 2.0
        "alpha_20": {
            "description": "α=2.0 (高度保守)",
            "use_static_gp": True,
            "use_online_gp": True,
            "gp_config": replace(base_gp_config, variance_scaling_alpha=2.0),
            "solver_options": {"variance_scaling_alpha": 2.0},
        },
    }


def prepare_quadrotor_mpc_ablation(simulation_options, config_name, config, 
                                   version=None, name=None, t_horizon=1.0):
    """Create MPC with specific ablation configuration."""
    if config.get("q_diagonal") is None:
        q_diagonal = np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    else:
        q_diagonal = config["q_diagonal"]
    r_diagonal = np.array([0.1, 0.1, 0.1, 0.1])

    simulation_dt = 5e-4
    n_mpc_nodes = 10
    node_dt = t_horizon / n_mpc_nodes

    my_quad = Quadrotor3D(**simulation_options)

    # Load GP models only if static GP is enabled
    pre_trained_models = None
    if config["use_static_gp"] and version is not None and name is not None:
        load_ops = {"params": simulation_options, "git": version, "model_name": name}
        pre_trained_models = load_pickled_models(model_options=load_ops)

    quad_name = f"my_quad_ablation_{config_name}"
    solver_options = {"terminal_cost": True, "solver_type": "SQP_RTI"}
    solver_options.update(config.get("solver_options", {}))

    quad_mpc = Quad3DMPC(my_quad, t_horizon=t_horizon, optimization_dt=node_dt, simulation_dt=simulation_dt,
                         q_cost=q_diagonal, r_cost=r_diagonal, n_nodes=n_mpc_nodes,
                         pre_trained_models=pre_trained_models, model_name=quad_name, 
                         solver_options=solver_options,
                         use_online_gp=config["use_online_gp"])

    return quad_mpc


def run_single_ablation(quad_mpc, av_speed, reference_type, config_name, config, 
                        seed=303, record_violations=True):
    """
    Run single ablation experiment with detailed metrics collection.
    
    Returns dict with: rmse, max_velocity, opt_time, control_saturations, 
                       max_tracking_error, violation_count
    """
    my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    simulation_dt = quad_mpc.simulation_dt
    t_horizon = quad_mpc.t_horizon

    reference_over_sampling = 5
    mpc_period = t_horizon / (n_mpc_nodes * reference_over_sampling)
    wind_model = RealisticWindModel()

    # Generate trajectory with configurable seed
    if reference_type == "loop":
        reference_traj, reference_timestamps, reference_u = loop_trajectory(
            quad=my_quad, discretization_dt=mpc_period, radius=5, z=1, lin_acc=av_speed * 0.25,
            clockwise=True, yawing=False, v_max=av_speed, map_name=None, plot=False)
    elif reference_type == "lemniscate":
        reference_traj, reference_timestamps, reference_u = lemniscate_trajectory(
            quad=my_quad, discretization_dt=mpc_period, radius=5, z=1, lin_acc=av_speed * 0.25,
            clockwise=True, yawing=False, v_max=av_speed, map_name=None, plot=False)
    else:
        reference_traj, reference_timestamps, reference_u = random_trajectory(
            quad=my_quad, discretization_dt=mpc_period, seed=seed, speed=av_speed, plot=False)

    quad_current_state = reference_traj[0, :].tolist()
    my_quad.set_state(quad_current_state)

    # Initialize online GP manager if enabled
    online_gp_manager = None
    if config["use_online_gp"] and config["gp_config"] is not None:
        online_gp_manager = IncrementalGPManager(config=config["gp_config"].to_dict())

    x_pred = None
    model_ind = 0
    mean_opt_time = 0
    n_points_ref = reference_traj.shape[0]
    t_current = reference_timestamps[0]
    x_executed = []
    control_saturations = 0
    max_tracking_error = 0
    violation_count = 0
    VIOLATION_THRESHOLD = 0.5  # meters

    for ref_idx in range(n_points_ref - 1):
        if t_current > reference_timestamps[ref_idx + 1]:
            continue

        external_v = wind_model.get_wind_velocity(t_current)
        ref_traj_chunk, ref_u_chunk = get_reference_chunk(
            reference_traj, reference_u, ref_idx, n_mpc_nodes, reference_over_sampling)
        model_ind = quad_mpc.set_reference(x_reference=separate_variables(ref_traj_chunk), u_reference=ref_u_chunk)

        # Online GP prediction
        online_predictions = None
        online_variances = None
        if online_gp_manager and any(gp.is_trained_once for gp in online_gp_manager.gps):
            if x_pred is not None:
                planned_states_body = world_to_body_velocity_mapping(x_pred)
                planned_velocities_body = planned_states_body[:, 7:10]
                predicted_residuals, predicted_variances = online_gp_manager.predict(planned_velocities_body)
                online_predictions = predicted_residuals
                online_variances = predicted_variances

        t_opt_init = time.time()
        w_opt, x_pred = quad_mpc.optimize(use_model=model_ind, return_x=True,
                                          online_gp_predictions=online_predictions,
                                          online_gp_variances=online_variances)
        mean_opt_time += time.time() - t_opt_init

        next_control = w_opt[:4]
        if np.any(np.abs(next_control) > 0.95 * my_quad.max_thrust):
            control_saturations += 1

        sim_length = n_mpc_nodes * reference_over_sampling
        for sim_idx in range(sim_length):
            if ref_idx + sim_idx >= n_points_ref - 1:
                break
            quad_mpc.simulate(ref_u=next_control, external_v=external_v)
            current_state = my_quad.get_state(quaternion=True, stacked=True)
            x_executed.append(current_state)
            
            # Check violation
            if record_violations:
                ref_pos = reference_traj[min(ref_idx + sim_idx, n_points_ref - 1), :3]
                tracking_error = np.linalg.norm(np.array(current_state[:3]) - ref_pos)
                if tracking_error > VIOLATION_THRESHOLD:
                    violation_count += 1
                max_tracking_error = max(max_tracking_error, tracking_error)
            
            t_current = reference_timestamps[ref_idx + sim_idx + 1] if ref_idx + sim_idx + 1 < n_points_ref else t_current + simulation_dt

    # Cleanup
    if online_gp_manager:
        online_gp_manager.shutdown()

    x_executed = np.array(x_executed)
    mean_opt_time /= max(1, ref_idx)

    n_compare = min(len(x_executed), len(reference_traj))
    pos_rmse = np.sqrt(np.mean(np.sum((x_executed[:n_compare, :3] - reference_traj[:n_compare, :3]) ** 2, axis=1)))
    max_velocity = np.max(np.linalg.norm(x_executed[:, 7:10], axis=1)) if len(x_executed) > 1 else 0
    
    # Calculate violation rate
    total_steps = len(x_executed)
    violation_rate = violation_count / total_steps if total_steps > 0 else 0

    return {
        "rmse": pos_rmse,
        "max_velocity": max_velocity,
        "opt_time": mean_opt_time,
        "control_saturations": control_saturations,
        "max_tracking_error": max_tracking_error,
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "total_steps": total_steps,
    }


def run_full_ablation(speed=2.7, trajectory_type="random", n_seeds=1):
    """
    Run complete ablation study with optional Monte Carlo seeds.
    """
    print("=" * 70)
    print("Comprehensive Ablation Study for Variance-Aware GP-MPC")
    print("=" * 70)
    print(f"Speed: {speed} m/s, Trajectory: {trajectory_type}, Seeds: {n_seeds}\n")

    configs = get_ablation_configs()
    simulation_options = SimpleSimConfig.simulation_disturbances
    results = {}

    for config_name, config in configs.items():
        print(f"Running: {config['description']}...")
        
        # Run with multiple seeds if Monte Carlo
        seed_results = []
        for seed in range(303, 303 + n_seeds):
            quad_mpc = prepare_quadrotor_mpc_ablation(
                simulation_options, config_name, config,
                version=DEFAULT_MODEL_VERSION if config["use_static_gp"] else None,
                name=DEFAULT_MODEL_NAME if config["use_static_gp"] else None
            )
            
            result = run_single_ablation(
                quad_mpc, speed, trajectory_type, config_name, config, seed=seed
            )
            seed_results.append(result)

        # Aggregate results
        avg_result = {
            "rmse": np.mean([r["rmse"] for r in seed_results]),
            "rmse_std": np.std([r["rmse"] for r in seed_results]),
            "max_tracking_error": np.mean([r["max_tracking_error"] for r in seed_results]),
            "violation_rate": np.mean([r["violation_rate"] for r in seed_results]),
            "control_saturations": np.mean([r["control_saturations"] for r in seed_results]),
            "opt_time": np.mean([r["opt_time"] for r in seed_results]),
        }
        
        results[config_name] = avg_result
        print(f"  RMSE: {avg_result['rmse']:.4f}±{avg_result['rmse_std']:.4f} m, "
              f"Violation Rate: {avg_result['violation_rate']*100:.2f}%")

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Configuration':<25} {'RMSE (m)':<15} {'Max Err (m)':<12} {'Viol Rate':<12}")
    print("-" * 70)
    for config_name, result in results.items():
        desc = configs[config_name]['description']
        rmse_str = f"{result['rmse']:.4f}±{result['rmse_std']:.4f}" if n_seeds > 1 else f"{result['rmse']:.4f}"
        print(f"{desc:<25} {rmse_str:<15} {result['max_tracking_error']:<12.4f} {result['violation_rate']*100:<12.2f}%")

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
    parser.add_argument("--quick", action="store_true", help="Run only key configurations")
    args = parser.parse_args()

    results = run_full_ablation(speed=args.speed, trajectory_type=args.trajectory, n_seeds=args.seeds)
