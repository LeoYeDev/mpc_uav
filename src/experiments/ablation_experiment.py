"""
Ablation experiment for variance-aware GP-MPC.

Compares different variance scaling strategies:
1. No GP (Nominal MPC)
2. GP only (no variance scaling)
3. GP + Variance Scaling (confidence-based)

Run: python src/experiments/ablation_experiment.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

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
# Ablation Configuration (Simplified: only variance_scaling_alpha)
# ============================================================================
ABLATION_CONFIGS = {
    "nominal": {
        "description": "Nominal MPC (无GP)",
        "use_online_gp": False,
        "use_offline_gp": False,
        "solver_options": {},
    },
    "gp_no_variance": {
        "description": "GP-MPC (α=0, 完全信任GP)",
        "use_online_gp": True,
        "use_offline_gp": True,
        "solver_options": {
            "variance_scaling_alpha": 0.0,  # 完全信任GP
        },
    },
    "gp_variance_low": {
        "description": "GP-MPC (α=0.5, 轻度保守)",
        "use_online_gp": True,
        "use_offline_gp": True,
        "solver_options": {
            "variance_scaling_alpha": 0.5,
        },
    },
    "gp_variance_default": {
        "description": "GP-MPC (α=1.0, 默认保守)",
        "use_online_gp": True,
        "use_offline_gp": True,
        "solver_options": {
            "variance_scaling_alpha": 1.0,
        },
    },
    "gp_variance_high": {
        "description": "GP-MPC (α=2.0, 高度保守)",
        "use_online_gp": True,
        "use_offline_gp": True,
        "solver_options": {
            "variance_scaling_alpha": 2.0,
        },
    },
}


def prepare_quadrotor_mpc_ablation(simulation_options, config_name, version=None, name=None, 
                                     reg_type="gp", t_horizon=1.0, q_diagonal=None, r_diagonal=None):
    """Create MPC with ablation configuration."""
    ablation_config = ABLATION_CONFIGS[config_name]
    
    if q_diagonal is None:
        q_diagonal = np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    if r_diagonal is None:
        r_diagonal = np.array([0.1, 0.1, 0.1, 0.1])

    simulation_dt = 5e-4
    n_mpc_nodes = 10
    node_dt = t_horizon / n_mpc_nodes

    my_quad = Quadrotor3D(**simulation_options)

    # Load GP models if needed
    if ablation_config["use_offline_gp"] and version is not None and name is not None:
        load_ops = {"params": simulation_options, "git": version, "model_name": name}
        pre_trained_models = load_pickled_models(model_options=load_ops)
    else:
        pre_trained_models = None

    quad_name = f"my_quad_ablation_{config_name}"
    solver_options = {"terminal_cost": True, "solver_type": "SQP_RTI"}
    solver_options.update(ablation_config["solver_options"])

    quad_mpc = Quad3DMPC(my_quad, t_horizon=t_horizon, optimization_dt=node_dt, simulation_dt=simulation_dt,
                         q_cost=q_diagonal, r_cost=r_diagonal, n_nodes=n_mpc_nodes,
                         pre_trained_models=pre_trained_models, model_name=quad_name, 
                         solver_options=solver_options,
                         use_online_gp=ablation_config["use_online_gp"])

    return quad_mpc


def run_ablation_experiment(quad_mpc, av_speed, reference_type="random", config_name="", 
                            online_gp_manager=None):
    """Run single ablation experiment."""
    my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    simulation_dt = quad_mpc.simulation_dt
    t_horizon = quad_mpc.t_horizon

    reference_over_sampling = 5
    mpc_period = t_horizon / (n_mpc_nodes * reference_over_sampling)
    wind_model = RealisticWindModel()

    # Generate trajectory
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
            quad=my_quad, discretization_dt=mpc_period, seed=303, speed=av_speed, plot=False)

    quad_current_state = reference_traj[0, :].tolist()
    my_quad.set_state(quad_current_state)

    x_pred = None
    model_ind = 0
    mean_opt_time = 0
    n_points_ref = reference_traj.shape[0]
    t_current = reference_timestamps[0]
    x_executed = []
    control_saturations = 0
    max_tracking_error = 0

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
            x_executed.append(my_quad.get_state(quaternion=True, stacked=True))
            t_current = reference_timestamps[ref_idx + sim_idx + 1] if ref_idx + sim_idx + 1 < n_points_ref else t_current + simulation_dt

        if len(x_executed) > 0:
            current_error = np.linalg.norm(np.array(x_executed[-1][:3]) - reference_traj[min(ref_idx + sim_length, n_points_ref - 1), :3])
            max_tracking_error = max(max_tracking_error, current_error)

    x_executed = np.array(x_executed)
    mean_opt_time /= max(1, ref_idx)

    n_compare = min(len(x_executed), len(reference_traj))
    pos_rmse = np.sqrt(np.mean(np.sum((x_executed[:n_compare, :3] - reference_traj[:n_compare, :3]) ** 2, axis=1)))
    max_velocity = np.max(np.linalg.norm(x_executed[:, 7:10], axis=1)) if len(x_executed) > 1 else 0

    return {
        "rmse": pos_rmse,
        "max_velocity": max_velocity,
        "opt_time": mean_opt_time,
        "control_saturations": control_saturations,
        "max_tracking_error": max_tracking_error,
    }


def run_full_ablation(speed=2.7, trajectory_type="random"):
    """Run ablation study comparing all configurations."""
    print("=" * 60)
    print("Variance-Aware GP-MPC Ablation Study")
    print("Mechanism: confidence = 1/(1 + α*variance)")
    print("=" * 60)
    print(f"Speed: {speed} m/s, Trajectory: {trajectory_type}\n")

    results = {}
    simulation_options = SimpleSimConfig.simulation_disturbances

    for config_name, config in ABLATION_CONFIGS.items():
        print(f"Running: {config['description']}...")
        
        quad_mpc = prepare_quadrotor_mpc_ablation(
            simulation_options, config_name,
            version=DEFAULT_MODEL_VERSION, name=DEFAULT_MODEL_NAME
        )

        online_gp_manager = None
        if config["use_online_gp"]:
            online_gp_manager = IncrementalGPManager(config=OnlineGPConfig().to_dict())

        result = run_ablation_experiment(
            quad_mpc, speed, trajectory_type, config_name, online_gp_manager
        )

        if online_gp_manager:
            online_gp_manager.shutdown()

        results[config_name] = result
        alpha = config["solver_options"].get("variance_scaling_alpha", "N/A")
        print(f"  α={alpha}, RMSE: {result['rmse']:.4f} m, Max Error: {result['max_tracking_error']:.4f} m")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Configuration':<30} {'α':<6} {'RMSE (m)':<12} {'Max Err (m)':<12}")
    print("-" * 60)
    for config_name, result in results.items():
        desc = ABLATION_CONFIGS[config_name]['description']
        alpha = ABLATION_CONFIGS[config_name]["solver_options"].get("variance_scaling_alpha", "-")
        print(f"{desc:<30} {alpha:<6} {result['rmse']:<12.4f} {result['max_tracking_error']:<12.4f}")

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
    args = parser.parse_args()

    results = run_full_ablation(speed=args.speed, trajectory_type=args.trajectory)
