""" Runs the experimental setup to compare different data-learned models for the MPC on the Simplified Simulator.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import argparse
import time
import csv
import os
import random
import numpy as np

from config.configuration_parameters import SimpleSimConfig
from config.gp_config import build_online_gp_config
from config.paths import DEFAULT_MODEL_VERSION, DEFAULT_MODEL_NAME, DEFAULT_MODEL_TYPE
from src.core.controller import Quad3DMPC
from src.core.dynamics import Quadrotor3D
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.utils import load_pickled_models, interpol_mse, separate_variables
from src.utils.wind_model import RealisticWindModel
from src.utils.trajectories import random_trajectory, lemniscate_trajectory, loop_trajectory
from src.visualization.plotting import (
    initialize_drone_plotter, draw_drone_simulation,
    get_experiment_files, mse_tracking_experiment_plot
)
from src.visualization.paper_plots import plot_tracking_error_comparison
from src.visualization.gp_online import visualize_gp_snapshot
from src.gp.rdrv import load_rdrv
from src.gp.utils import world_to_body_velocity_mapping
from src.gp.online import IncrementalGPManager
from src.utils.data_logger import DataLogger
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def prepare_quadrotor_mpc(simulation_options, version=None, name=None, reg_type="gp", quad_name=None,
                          t_horizon=1.0, q_diagonal=None, r_diagonal=None, q_mask=None,
                          use_online_gp=False, solver_options=None):
    """
    Creates a Quad3DMPC for the custom simulator.
    @param simulation_options: Parameters for the Quadrotor3D object.
    @param version: loading version for the GP/RDRv model.
    @param name: name to load for the GP/RDRv model.
    @param reg_type: either `gp` or `rdrv`.
    @param quad_name: Name for the quadrotor. Default name will be used if not specified.
    @param t_horizon: Time horizon of MPC in seconds.
    @param q_diagonal: 12-dimensional diagonal of the Q matrix (p_xyz, a_xyz, v_xyz, w_xyz)
    @param r_diagonal: 4-dimensional diagonal of the R matrix (motor inputs 1-4)
    @param q_mask: State variable weighting mask (boolean). Which state variables compute towards state loss function?
    @param solver_options: Optional dictionary of solver options (e.g. variance_scaling_alpha).

    @return: A Quad3DMPC wrapper for the custom simulator.
    @rtype: Quad3DMPC
    """

    # Default Q and R matrix for LQR cost
    if q_diagonal is None:
        q_diagonal = np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    if r_diagonal is None:
        r_diagonal = np.array([0.1, 0.1, 0.1, 0.1])
    if q_mask is None:
        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

    # Simulation integration step (the smaller the more "continuous"-like simulation.
    simulation_dt = 5e-4

    # Number of MPC optimization nodes
    n_mpc_nodes = 10

    # Calculate time between two MPC optimization nodes [s]
    node_dt = t_horizon / n_mpc_nodes

    # Quadrotor simulator
    my_quad = Quadrotor3D(**simulation_options)

    if version is not None and name is not None:

        load_ops = {"params": simulation_options}
        load_ops.update({"git": version, "model_name": name})

        # Load trained GP model
        if reg_type == "gp":
            pre_trained_models = load_pickled_models(model_options=load_ops)
            rdrv_d = None

        else:
            rdrv_d = load_rdrv(model_options=load_ops)
            pre_trained_models = None

    else:
        pre_trained_models = rdrv_d = None

    if quad_name is None:
        # Add unique suffix based on online GP usage to avoid ACADOS solver cache conflicts
        suffix = "_ar" if use_online_gp else "_sgp" if version is not None else "_nom"
        quad_name = "my_quad_sim" + suffix

    # Initialize quad MPC
    quad_mpc = Quad3DMPC(my_quad, t_horizon=t_horizon, optimization_dt=node_dt, simulation_dt=simulation_dt,
                         q_cost=q_diagonal, r_cost=r_diagonal, n_nodes=n_mpc_nodes,
                         pre_trained_models=pre_trained_models, model_name=quad_name, q_mask=q_mask, rdrv_d_mat=rdrv_d,
                         use_online_gp=use_online_gp, solver_options=solver_options)

    return quad_mpc



def main(quad_mpc, av_speed, reference_type=None, plot=False,
         use_offline_gp=False, use_online_gp=False, 
         online_gp_manager=None, model_label="nominal",
         wind_profile="default", trajectory_seed=303,
         online_gp_pred_points=None,
         step_metrics_csv_path=None,
         buffer_debug_interval=0,
         max_steps=None):
    """
    Run tracking experiment with unified configuration.

    :param quad_mpc: Quad3DMPC controller instance
    :param av_speed: Average speed for trajectory
    :param reference_type: Trajectory type
    :param plot: Whether to plot real-time simulation
    :param use_offline_gp: Whether offline GP is active (SGP)
    :param use_online_gp: Whether online GP is active (AR)
    :param online_gp_manager: Manager for online GP training
    :param model_label: Label for result plotting and logging
    :param online_gp_pred_points: 仅用于快照绘图的最大预测点数量（控制补偿始终使用 N）
    :param step_metrics_csv_path: 每步控制统计CSV路径（None 表示不导出）
    :param buffer_debug_interval: buffer 调试打印间隔（步）；0 表示关闭
    :param max_steps: 最大控制步数；None 表示按整段轨迹运行
    """
    # 固定随机源：同一 seed 下保证风场、GP 初始化与轨迹采样可复现。
    seed_int = int(trajectory_seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    if torch is not None:
        torch.manual_seed(seed_int)

    # Recover some necessary variables from the MPC object
    my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    simulation_dt = quad_mpc.simulation_dt
    t_horizon = quad_mpc.t_horizon

    reference_over_sampling = 5
    mpc_period = t_horizon / (n_mpc_nodes * reference_over_sampling)
    #预测时长1s，一周期内点数为10，启用mpc周期0.02s

    # Initialize Wind Model (always used)
    wind_model = RealisticWindModel(profile=wind_profile)

    # Choose the reference trajectory:
    if reference_type == "loop":
        # Circular trajectory
        reference_traj, reference_timestamps, reference_u = loop_trajectory(
            quad=my_quad, discretization_dt=mpc_period, radius=5, z=1, lin_acc=av_speed * 0.25, clockwise=True,
            yawing=False, v_max=av_speed, map_name=None, plot=plot)
    elif reference_type == "lemniscate":
        # Lemniscate trajectory
        reference_traj, reference_timestamps, reference_u = lemniscate_trajectory(
            quad=my_quad, discretization_dt=mpc_period, radius=5, z=1, lin_acc=av_speed * 0.25, clockwise=True,
            yawing=False, v_max=av_speed, map_name=None, plot=plot)
    else:
        # Get a random smooth position trajectory
        reference_traj, reference_timestamps, reference_u = random_trajectory(
            quad=my_quad, discretization_dt=mpc_period, seed=trajectory_seed, speed=av_speed, plot=plot)

    max_available_steps = max(0, int(reference_traj.shape[0] - 1))
    if max_steps is None:
        max_control_steps = int(max_available_steps)
    else:
        max_control_steps = max(0, min(int(max_steps), int(max_available_steps)))

    # Set quad initial state equal to the initial reference trajectory state
    quad_current_state = reference_traj[0, :].tolist()
    my_quad.set_state(quad_current_state)

    real_time_artists = None
    if plot:
        # Initialize real time plot stuff
        world_radius = np.max(np.abs(reference_traj[:, :2])) * 1.2
        real_time_artists = initialize_drone_plotter(n_props=n_mpc_nodes, quad_rad=my_quad.length,
                                                     world_rad=world_radius, full_traj=reference_traj)

    start_time = time.time()
    max_simulation_time = 20000

    ref_u = reference_u[0, :]
    trajectory_len = int(max_control_steps + 1)
    quad_trajectory = np.zeros((trajectory_len, len(quad_current_state)))
    u_optimized_seq = np.zeros((trajectory_len, 4))

    # Sliding reference trajectory initial index
    current_idx = 0

    # Latency accounting (统一口径：不含sleep/仿真推进/绘图):
    # 1) 求解器耗时
    # 2) 在线GP预测耗时
    # 3) 在线更新+队列开销
    # 4) 完整控制计算耗时（1+2+3+其余控制逻辑）
    mpc_opt_time_acc = 0.0
    gp_predict_time_acc = 0.0
    gp_update_time_acc = 0.0
    buffer_update_time_acc = 0.0
    queue_overhead_time_acc = 0.0
    control_compute_time_acc = 0.0
    control_pre_sim_time_acc = 0.0
    control_post_sim_time_acc = 0.0

    # Measure total simulation time
    total_sim_time = 0.0

    # Initialize the online GP manager and history for main
    simulation_time = 0.0

    out_online_gp_manager = None  # 用于快照可视化的在线GP管理器
    out_x_pred = None
    out_total_sim_time = 0.0
    out_snapshot_quality = None
    best_snapshot_score = -1.0
    x_pred = None

    # 运行时分解统计（在线GP管理器内部）
    runtime_stats_prev = (
        online_gp_manager.get_runtime_stats(reset=False)
        if (online_gp_manager is not None and hasattr(online_gp_manager, "get_runtime_stats"))
        else {}
    )
    collect_step_metrics = bool(step_metrics_csv_path)
    step_metrics_rows = []
    running_err_sum = 0.0

    # 方差是否参与在线补偿由 MPC 风险项决定（alpha>0 时启用）。
    alpha = float(getattr(quad_mpc.quad_opt, "variance_scaling_alpha", 0.0))
    need_variance_for_control = bool(abs(alpha) > 1e-12)

    while (time.time() - start_time) < max_simulation_time and current_idx < max_control_steps:
        iter_compute_start = time.perf_counter()

        quad_current_state = my_quad.get_state(quaternion=True, stacked=True)

        quad_trajectory[current_idx, :] = np.expand_dims(quad_current_state, axis=0)
        u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

        # ##### Optimization runtime (outer loop) ##### #
        # Get the chunk of trajectory required for the current optimization
        ref_traj_chunk, ref_u_chunk = get_reference_chunk(
            reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling)

        model_ind = quad_mpc.set_reference(x_reference=separate_variables(ref_traj_chunk), u_reference=ref_u_chunk)

        # ========================================================================
        # 在线GP预测
        # ========================================================================
        online_predictions = None
        online_variances = None
        predict_call_ms = 0.0
        
        # 在线补偿查询点严格与 MPC 阶段对齐：使用 x_pred[0:N]
        effective_query_velocities = None
        if (
            online_gp_manager
            and use_online_gp
            and x_pred is not None
            and any(gp.is_trained_once for gp in online_gp_manager.gps)
        ):
            planned_states_body = world_to_body_velocity_mapping(np.array(x_pred, copy=False))
            planned_velocities_body = np.array(planned_states_body[:, 7:10], dtype=float, copy=False)
            n_needed = int(n_mpc_nodes)

            if planned_velocities_body.shape[0] <= 0:
                effective_query_velocities = np.zeros((n_needed, 3), dtype=float)
            elif planned_velocities_body.shape[0] >= n_needed:
                effective_query_velocities = planned_velocities_body[:n_needed, :]
            else:
                # 若预测轨迹长度不足 N，使用最后一个可用速度补齐，保证与 MPC 阶段一一匹配。
                tail = np.tile(planned_velocities_body[-1, :], (n_needed - planned_velocities_body.shape[0], 1))
                effective_query_velocities = np.vstack((planned_velocities_body, tail))

            t_predict_start = time.perf_counter()
            predicted_residuals, predicted_variances = online_gp_manager.predict(
                effective_query_velocities,
                need_variance=need_variance_for_control,
                clamp_extrapolation=False,
            )
            predict_call_ms = (time.perf_counter() - t_predict_start) * 1000.0
            online_predictions = predicted_residuals
            online_variances = predicted_variances if need_variance_for_control else None
        # ========================================================================

        # Optimize control input to reach pre-set target
        t_opt_init = time.perf_counter()
        w_opt, x_pred = quad_mpc.optimize(use_model=model_ind, return_x=True, 
                                          online_gp_predictions=online_predictions,
                                          online_gp_variances=online_variances)
        solver_step_time = time.perf_counter() - t_opt_init
        mpc_opt_time_acc += solver_step_time
        gp_predict_time_acc += predict_call_ms / 1000.0

        # Select first input (one for each motor) - MPC applies only first optimized input to the plant
        ref_u = np.squeeze(np.array(w_opt[:4]))

        # 可视化快照候选：优先“部分内插 + 部分外推”且训练集覆盖质量较高的时刻。
        if (
            online_gp_manager and use_online_gp and x_pred is not None
            and any(gp.is_trained_once for gp in online_gp_manager.gps)
            and effective_query_velocities is not None
            and plot
            and current_idx >= max(30, 2 * int(n_mpc_nodes))
        ):
            try:
                future_velocities = np.array(effective_query_velocities, copy=False)
                dim_mix_scores = []
                dim_span_scores = []
                dim_in_ratios = []
                valid_dim_count = 0
                mixed_dim_count = 0
                runtime_for_snapshot = (
                    online_gp_manager.get_runtime_stats(reset=False)
                    if hasattr(online_gp_manager, "get_runtime_stats")
                    else {}
                )
                unique_ratio_snapshot = float(runtime_for_snapshot.get("unique_ratio_mean_last", np.nan))
                selected_size_snapshot = float(runtime_for_snapshot.get("selected_size_mean_last", np.nan))
                buffer_cap = float(getattr(online_gp_manager, "buffer_max_size", 0.0))
                selected_fill_ratio = (
                    float(selected_size_snapshot / buffer_cap)
                    if (buffer_cap > 1e-9 and np.isfinite(selected_size_snapshot))
                    else 0.0
                )
                for dim_idx, gp in enumerate(online_gp_manager.gps):
                    if hasattr(gp.buffer, "get_effective_size_fast"):
                        n_train = int(gp.buffer.get_effective_size_fast())
                    else:
                        n_train = len(gp.buffer.get_training_set())
                    if n_train < 8:
                        continue
                    if hasattr(gp.buffer, "get_velocity_bounds_fast"):
                        v_min, v_max = gp.buffer.get_velocity_bounds_fast()
                    else:
                        train_data = gp.buffer.get_training_set()
                        if len(train_data) < 2:
                            continue
                        vx = np.array([p[0] for p in train_data], dtype=float)
                        v_min, v_max = float(np.min(vx)), float(np.max(vx))
                    if v_min is None or v_max is None:
                        continue
                    qv = future_velocities[:, dim_idx]
                    span = max(v_max - v_min, 1e-6)
                    margin = 0.10 * span
                    in_range = (qv >= (v_min - margin)) & (qv <= (v_max + margin))
                    in_ratio = float(np.mean(in_range))
                    valid_dim_count += 1
                    dim_in_ratios.append(in_ratio)
                    # 目标是“内插与外推并存”，希望比例在中间区间。
                    target_ratio = 0.70
                    mix_score = max(0.0, 1.0 - abs(in_ratio - target_ratio) / target_ratio)
                    if 0.20 <= in_ratio <= 0.85:
                        mixed_dim_count += 1
                    dim_mix_scores.append(mix_score)
                    dim_span_scores.append(float(np.max(qv) - np.min(qv)))

                if valid_dim_count > 0:
                    mixed_dim_ratio = mixed_dim_count / float(valid_dim_count)
                    mean_pred_span = float(np.mean(dim_span_scores))
                    if mean_pred_span < 0.30:
                        # 预测速度范围过窄时不作为快照候选，避免图像“扎堆”。
                        raise RuntimeError("snapshot candidate skipped: predicted span too narrow")
                    score = (
                        2.2 * mixed_dim_ratio
                        + float(np.mean(dim_mix_scores))
                        + 0.15 * np.log1p(mean_pred_span)
                        + 0.55 * (0.0 if not np.isfinite(unique_ratio_snapshot) else np.clip(unique_ratio_snapshot, 0.0, 1.0))
                        + 0.25 * np.clip(selected_fill_ratio, 0.0, 1.0)
                    )
                    # 希望保留一定外推占比，但避免几乎全外推。
                    out_ratio = float(1.0 - np.mean(dim_in_ratios))
                    if 0.10 <= out_ratio <= 0.55:
                        score += 0.35
                    if score >= best_snapshot_score:
                        best_snapshot_score = score
                        out_online_gp_manager = online_gp_manager
                        out_x_pred = np.array(x_pred, copy=True)
                        out_total_sim_time = float(total_sim_time)
                        out_snapshot_quality = {
                            "mixed_dim_ratio": float(mixed_dim_ratio),
                            "mean_in_ratio": float(np.mean(dim_in_ratios)),
                            "mean_pred_span": mean_pred_span,
                            "pred_points_used": int(future_velocities.shape[0]),
                            "unique_ratio": float(unique_ratio_snapshot) if np.isfinite(unique_ratio_snapshot) else np.nan,
                            "selected_fill_ratio": float(selected_fill_ratio),
                            "out_of_range_ratio": float(1.0 - np.mean(dim_in_ratios)),
                        }
            except Exception:
                pass

        simulation_time = 0.0
        
        # --- ADDED: 在线GP输入速度（用于更新）与可视化采样 ---
        if online_gp_manager and use_online_gp: 
            s_before_sim  = quad_mpc.get_state()
            v_body_in = s_before_sim.T
            v_body_in = world_to_body_velocity_mapping(v_body_in)
            v_body_in = np.squeeze(v_body_in[:,7:10])  # Extract only the velocity components
        # --- END ADDED:

        # --- 核心修改：基于无人机完整状态计算风力 ---
        # 直接将当前13维状态向量传入 (时间)
        ext_v_k = wind_model.get_wind_velocity(total_sim_time)
        # ---------------------------------------------
        pre_sim_elapsed = time.perf_counter() - iter_compute_start
        control_compute_time_acc += pre_sim_elapsed
        control_pre_sim_time_acc += pre_sim_elapsed

        # 可视化绘制不计入控制计算时延统计。
        if len(quad_trajectory) > 0 and plot and current_idx > 0:
            draw_drone_simulation(real_time_artists, quad_trajectory[:current_idx, :], my_quad, targets=None,
                                  targets_reached=None, pred_traj=x_pred, x_pred_cov=None)

        # ##### Simulation runtime (inner loop) ##### #
        while simulation_time < mpc_period:
            simulation_time += simulation_dt
            total_sim_time += simulation_dt
            quad_mpc.simulate(ref_u, external_v=ext_v_k)


        # --- ADDED: 在线GP的数据收集与异步更新
        post_sim_start = time.perf_counter()
        step_buffer_update_ms = 0.0
        step_queue_overhead_ms = 0.0
        selected_size_mean = float(runtime_stats_prev.get("selected_size_mean_last", np.nan)) if runtime_stats_prev else np.nan
        unique_ratio_mean = float(runtime_stats_prev.get("unique_ratio_mean_last", np.nan)) if runtime_stats_prev else np.nan
        full_merge_calls = int(runtime_stats_prev.get("full_merge_calls", 0)) if runtime_stats_prev else 0
        if online_gp_manager and use_online_gp: 
            #推演后的状态
            s_after_sim  = quad_mpc.get_state()
            v_body_out = s_after_sim.T
            v_body_out = world_to_body_velocity_mapping(v_body_out)
            v_body_out = np.squeeze(v_body_out[:,7:10])  # Extract only the velocity components

            # x_predic 
            v_body_predic = quad_mpc.predict_model_step_accurately(
                    current_state_np=s_before_sim,
                    control_input_np=w_opt[:4],
                    integration_period=simulation_time,
                    use_model_idx=model_ind
                )
            v_body_predic = np.expand_dims(v_body_predic, axis=0)
            v_body_predic = world_to_body_velocity_mapping(v_body_predic)
            v_body_predic = np.squeeze(v_body_predic[:,7:10])   # Extract only the velocity components

            residual_acc_body = v_body_out - v_body_predic
            residual_acc_body /= simulation_time

            #更新在线GP
            update_start_time = time.perf_counter()

            # --- 异步更新与轮询 ---
            online_gp_manager.update(v_body_in, residual_acc_body, timestamp=total_sim_time)
            online_gp_manager.poll_for_results()

            gp_update_time_acc += time.perf_counter() - update_start_time
            if hasattr(online_gp_manager, "get_runtime_stats"):
                runtime_stats_curr = online_gp_manager.get_runtime_stats(reset=False)
                step_buffer_update_ms = max(
                    0.0,
                    float(runtime_stats_curr.get("buffer_update_ms", 0.0))
                    - float(runtime_stats_prev.get("buffer_update_ms", 0.0)),
                )
                step_queue_overhead_ms = max(
                    0.0,
                    float(runtime_stats_curr.get("queue_overhead_ms", 0.0))
                    - float(runtime_stats_prev.get("queue_overhead_ms", 0.0)),
                )
                selected_size_mean = float(runtime_stats_curr.get("selected_size_mean_last", np.nan))
                unique_ratio_mean = float(runtime_stats_curr.get("unique_ratio_mean_last", np.nan))
                full_merge_calls = int(runtime_stats_curr.get("full_merge_calls", 0))
                runtime_stats_prev = runtime_stats_curr
        # --- END ADDED: Online GP Initialization ---
        post_sim_elapsed = time.perf_counter() - post_sim_start
        control_compute_time_acc += post_sim_elapsed
        control_post_sim_time_acc += post_sim_elapsed
        buffer_update_time_acc += step_buffer_update_ms / 1000.0
        queue_overhead_time_acc += step_queue_overhead_ms / 1000.0

        if collect_step_metrics:
            # 运行中 RMSE（当前步，便于诊断策略收敛过程）
            sim_state_after = quad_mpc.get_state().reshape(-1)
            ref_idx = min(current_idx + 1, reference_traj.shape[0] - 1)
            err_now = float(np.sqrt(np.sum((sim_state_after[:3] - reference_traj[ref_idx, :3]) ** 2)))
            running_err_sum += err_now
            running_rmse = running_err_sum / max(current_idx + 1, 1)

            step_metrics_rows.append({
                "step": int(current_idx),
                "sim_time_s": float(total_sim_time),
                "solver_ms": float(solver_step_time * 1000.0),
                "gp_predict_ms": float(predict_call_ms),
                "buffer_update_ms": float(step_buffer_update_ms),
                "queue_overhead_ms": float(step_queue_overhead_ms),
                "control_time_ms": float((pre_sim_elapsed + post_sim_elapsed) * 1000.0),
                "selected_size": float(selected_size_mean),
                "unique_ratio": float(unique_ratio_mean),
                "full_merge_calls": int(full_merge_calls),
                "rmse_running": float(running_rmse),
            })

        if (
            online_gp_manager is not None
            and use_online_gp
            and int(buffer_debug_interval) > 0
            and (current_idx % int(buffer_debug_interval) == 0)
        ):
            debug_chunks = []
            for dim_idx, gp in enumerate(online_gp_manager.gps):
                train_points = []
                if hasattr(gp.buffer, "get_training_set_full"):
                    train_points = gp.buffer.get_training_set_full()
                else:
                    train_points = gp.buffer.get_training_set()
                if len(train_points) == 0:
                    debug_chunks.append(f"d{dim_idx}: n=0")
                    continue
                v_arr = np.array([p[0] for p in train_points], dtype=float)
                diag = gp.buffer.get_diagnostics() if hasattr(gp.buffer, "get_diagnostics") else {}
                debug_chunks.append(
                    (
                        "d{d}: n={n}, v=[{vmin:.2f},{vmax:.2f}], std={std:.3f}, uniq={uniq:.2f}, "
                        "acc={acc:.2f}, skip={skip:.2f}, prune={prune:.0f}, flip={flip:.0f}"
                    ).format(
                        d=dim_idx,
                        n=len(train_points),
                        vmin=float(np.min(v_arr)),
                        vmax=float(np.max(v_arr)),
                        std=float(np.std(v_arr)),
                        uniq=float(diag.get("unique_ratio", np.nan)),
                        acc=float(diag.get("insert_accept_ratio", np.nan)),
                        skip=float(diag.get("insert_skip_ratio", np.nan)),
                        prune=float(diag.get("prune_old_count_last", np.nan)),
                        flip=float(diag.get("flip_delete_count_last", np.nan)),
                    )
                )
            print(
                f"[buffer-debug] step={current_idx}, t={total_sim_time:.2f}s | "
                + " | ".join(debug_chunks)
            )

        u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))
        current_idx += 1   

    quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
    final_idx = min(int(current_idx), int(quad_trajectory.shape[0] - 1))
    quad_trajectory[final_idx, :] = np.expand_dims(quad_current_state, axis=0)
    u_optimized_seq[final_idx, :] = np.reshape(ref_u, (1, -1))
    
    # Average elapsed time per control step
    mean_opt_time = mpc_opt_time_acc / max(current_idx, 1)
    mean_gp_predict_time = gp_predict_time_acc / max(current_idx, 1)
    mean_gp_update_time = gp_update_time_acc / max(current_idx, 1)
    mean_buffer_update_time = buffer_update_time_acc / max(current_idx, 1)
    mean_queue_overhead_time = queue_overhead_time_acc / max(current_idx, 1)
    mean_total_ctrl_time = control_compute_time_acc / max(current_idx, 1)
    mean_pre_sim_ctrl_time = control_pre_sim_time_acc / max(current_idx, 1)
    mean_post_sim_ctrl_time = control_post_sim_time_acc / max(current_idx, 1)

    eval_len = int(min(current_idx + 1, reference_traj.shape[0], quad_trajectory.shape[0], len(reference_timestamps)))
    rmse = interpol_mse(
        reference_timestamps[:eval_len],
        reference_traj[:eval_len, :3],
        reference_timestamps[:eval_len],
        quad_trajectory[:eval_len, :3],
    )
    max_vel = np.max(np.sqrt(np.sum(reference_traj[:eval_len, 7:10] ** 2, 1)))
    
    print(f'\n--- Simulation finished ---\n')
    print(f'Average optimization time (solve only): {mean_opt_time:.4f} s')
    print(f'Average online predict time: {mean_gp_predict_time:.4f} s')
    print(f'Average online update overhead: {mean_gp_update_time:.4f} s')
    print(f'Average buffer update overhead: {mean_buffer_update_time:.4f} s')
    print(f'Average queue overhead: {mean_queue_overhead_time:.4f} s')
    print(f'Average control compute time (full pipeline, no sleep/sim): {mean_total_ctrl_time:.4f} s')
    print(f'  - pre-sim compute: {mean_pre_sim_ctrl_time:.4f} s')
    print(f'  - post-sim compute: {mean_post_sim_ctrl_time:.4f} s')
    print(f'Evaluation length: {eval_len} states (max_steps={max_steps if max_steps is not None else "full"})')
    print(f'RMSE: {rmse:.4f} m')
    print(f'Maximum velocity: {max_vel:.2f} m/s')

    # 控制回路可观测性导出（CSV）
    if collect_step_metrics and len(step_metrics_rows) > 0:
        out_csv = os.path.abspath(step_metrics_csv_path)
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "sim_time_s",
                    "solver_ms",
                    "gp_predict_ms",
                    "buffer_update_ms",
                    "queue_overhead_ms",
                    "control_time_ms",
                    "selected_size",
                    "unique_ratio",
                    "full_merge_calls",
                    "rmse_running",
                ],
            )
            writer.writeheader()
            for row in step_metrics_rows:
                writer.writerow(row)
        print(f"Saved step metrics CSV: {out_csv}")

    if plot:
        if wind_model is not None:
            wind_model.visualize() # 在仿真开始前调用可视化
        
        if out_online_gp_manager is not None:
            if out_snapshot_quality is not None:
                print(
                    "[snapshot] selected time="
                    f"{out_total_sim_time:.2f}s, mixed_dim_ratio="
                    f"{out_snapshot_quality['mixed_dim_ratio']:.2f}, "
                    f"mean_in_ratio={out_snapshot_quality['mean_in_ratio']:.2f}, "
                    f"mean_pred_span={out_snapshot_quality['mean_pred_span']:.2f}, "
                    f"pred_points={out_snapshot_quality['pred_points_used']}"
                )
            visualize_gp_snapshot(
                online_gp_manager=out_online_gp_manager,
                # 使用当前的MPC预测轨迹来定义绘图的X轴范围
                mpc_planned_states=out_x_pred, 
                snapshot_info_str=f"In-Flight Snapshot @ SimTime {out_total_sim_time:.2f}s",
                max_pred_points=(
                    int(n_mpc_nodes)
                    if online_gp_pred_points is None
                    else int(max(1, min(int(online_gp_pred_points), int(n_mpc_nodes))))
                ),
            )
            #out_online_gp_manager.visualize_training_history()
        
    # --- 修改 1: 增加函数返回值，用于后续保存 ---
    runtime_details = {
        "solver_time": float(mean_opt_time),
        "gp_predict_time": float(mean_gp_predict_time),
        "gp_update_time": float(mean_gp_update_time),
        "buffer_update_time": float(mean_buffer_update_time),
        "queue_overhead_time": float(mean_queue_overhead_time),
        "control_time": float(mean_total_ctrl_time),
        "variance_scaling_alpha": float(alpha),
        "need_variance_for_control": bool(need_variance_for_control),
        "step_metrics_csv_path": os.path.abspath(step_metrics_csv_path) if step_metrics_csv_path else "",
        "online_gp_async_enabled": bool(getattr(online_gp_manager, "async_hp_updates", False)) if online_gp_manager else False,
        "online_gp_backend": str(getattr(online_gp_manager, "_worker_backend", "none")) if online_gp_manager else "none",
    }
    return (
        rmse,
        max_vel,
        mean_opt_time,
        reference_timestamps[:eval_len],
        reference_traj[:eval_len, :],
        quad_trajectory[:eval_len, :],
        mean_gp_update_time,
        mean_total_ctrl_time,
        runtime_details,
    )
    # --- 修改结束 ---


if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Comparative experiment: SGP vs AR-MPC")
    parser.add_argument("--controller", type=str, default="ar", choices=["ar", "sgp", "both"],
                        help="运行控制器类型：ar / sgp / both")
    parser.add_argument("--trajectory", type=str, default="random", choices=["random", "loop", "lemniscate"],
                        help="轨迹类型")
    parser.add_argument("--speed", type=float, default=3.0, help="平均速度 (m/s)")
    parser.add_argument("--wind-profile", type=str, default="default", choices=["default", "regime_shift"],
                        help="风场模式")
    parser.add_argument("--seed", type=int, default=303, help="随机轨迹种子")
    parser.add_argument("--plot", action="store_true", help="启用实时仿真绘图")
    parser.add_argument("--no-summary-plots", action="store_true", help="跳过最终对比图绘制")
    parser.add_argument("--step-metrics-csv", type=str, default="",
                        help="每步统计 CSV 输出路径；若 controller=both 会自动加后缀")
    parser.add_argument("--buffer-debug-interval", type=int, default=0,
                        help="在线 buffer 调试打印步长（0=关闭）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="最大控制步数（默认按整段轨迹运行）")
    args = parser.parse_args()

    traj_type_vec = [args.trajectory]
    traj_type_labels = [args.trajectory.capitalize()]
    av_speed_vec = [[float(args.speed)]]

    git_list = DEFAULT_MODEL_VERSION
    name_list = DEFAULT_MODEL_NAME
    type_list = DEFAULT_MODEL_TYPE
    plot_sim = bool(args.plot)
    noisy_sim_options = SimpleSimConfig.simulation_disturbances

    model_vec = []
    legends = []
    if args.controller in ("ar", "both"):
        model_vec.append({
            "simulation_options": noisy_sim_options,
            "model": {"version": git_list, "name": name_list, "reg_type": type_list, "use_online_gp": True},
            "description": "AR-MPC",
        })
        legends.append("AR")
    if args.controller in ("sgp", "both"):
        model_vec.append({
            "simulation_options": noisy_sim_options,
            "model": {"version": git_list, "name": name_list, "reg_type": type_list, "use_online_gp": False},
            "description": "SGP-MPC",
        })
        legends.append("SGP")

    print("==============================================================")
    print(f"Comparative Run | controller={args.controller}, traj={args.trajectory}, speed={args.speed:.2f}, wind={args.wind_profile}")
    print("==============================================================")

    experiment_logger = DataLogger()
    y_label = "RMSE [m]"
    mse = np.zeros((len(traj_type_vec), len(av_speed_vec[0]), len(model_vec)))
    v_max = np.zeros((len(traj_type_vec), len(av_speed_vec[0])))
    t_opt = np.zeros((len(traj_type_vec), len(av_speed_vec[0]), len(model_vec)))

    for n_train_id, model_type in enumerate(model_vec):
        online_gp_manager = None
        use_online_gp = bool(model_type["model"] and model_type["model"].get("use_online_gp", False))
        if use_online_gp:
            # 统一从 gp_config 派生，避免在脚本中硬编码在线GP超参数。
            online_cfg = build_online_gp_config()
            online_gp_manager = IncrementalGPManager(
                config=online_cfg.to_dict()
            )
            print(
                f"[online-gp] async_enabled={online_gp_manager.async_hp_updates}, "
                f"backend={getattr(online_gp_manager, '_worker_backend', 'unknown')}"
            )

        custom_mpc = prepare_quadrotor_mpc(model_type["simulation_options"], **model_type["model"])
        use_offline_gp = True
        model_desc = model_type.get("description", legends[n_train_id])

        for traj_id, traj_type in enumerate(traj_type_vec):
            for v_id, speed in enumerate(av_speed_vec[traj_id]):
                traj_params = {"av_speed": speed, "reference_type": traj_type, "plot": plot_sim}
                if online_gp_manager:
                    online_gp_manager.reset()

                step_csv = None
                if args.step_metrics_csv:
                    base, ext = os.path.splitext(args.step_metrics_csv)
                    if not ext:
                        ext = ".csv"
                    if args.controller == "both":
                        step_csv = f"{base}_{legends[n_train_id].lower()}{ext}"
                    else:
                        step_csv = f"{base}{ext}"

                run_result = main(
                    custom_mpc,
                    **traj_params,
                    use_online_gp=use_online_gp,
                    use_offline_gp=use_offline_gp,
                    model_label=model_desc,
                    online_gp_manager=online_gp_manager,
                    wind_profile=args.wind_profile,
                    trajectory_seed=int(args.seed),
                    step_metrics_csv_path=step_csv,
                    buffer_debug_interval=int(args.buffer_debug_interval),
                    max_steps=args.max_steps,
                )
                (
                    mse[traj_id, v_id, n_train_id],
                    traj_v,
                    _opt_dt,
                    t_ref,
                    x_ref,
                    x_executed,
                    _mean_gp_update_dt,
                    mean_ctrl_dt,
                    runtime_details,
                ) = run_result

                t_opt[traj_id, v_id, n_train_id] += mean_ctrl_dt
                if v_max[traj_id, v_id] == 0:
                    v_max[traj_id, v_id] = traj_v

                print(
                    f"[result] {model_desc}: RMSE={mse[traj_id, v_id, n_train_id]:.4f} m, "
                    f"control={mean_ctrl_dt*1000.0:.2f} ms, "
                    f"solver={runtime_details.get('solver_time', np.nan)*1000.0:.2f} ms, "
                    f"gp_predict={runtime_details.get('gp_predict_time', 0.0)*1000.0:.2f} ms, "
                    f"gp_update={runtime_details.get('gp_update_time', 0.0)*1000.0:.2f} ms"
                )

                if v_id == len(av_speed_vec[traj_id]) - 1:
                    experiment_logger.log(legends[n_train_id], {
                        "t_ref": t_ref,
                        "x_ref": x_ref,
                        "x_executed": x_executed,
                    })

        if online_gp_manager:
            online_gp_manager.shutdown()

    _, err_file, v_file, t_file = get_experiment_files()
    np.save(err_file, mse)
    np.save(v_file, v_max)
    np.save(t_file, t_opt)

    if not args.no_summary_plots:
        controller_plot_map = {
            "AR": {"color": "#7C7CBA", "linestyle": "-", "linewidth": 1, "label": "AR-MPC", "fill_alpha": 0.25, "zorder": 4},
            "SGP": {"color": "#EECA40", "linestyle": "-", "linewidth": 1, "label": "SGP-MPC", "fill_alpha": 0.05, "zorder": 3},
            "nominal": {"color": "#23BAC5", "linestyle": "-", "linewidth": 1, "label": "Nominal MPC", "fill_alpha": 0.15, "zorder": 1},
            "perfect": {"color": "#2ecc71", "linestyle": "--", "linewidth": 1, "label": "Perfect Model", "fill_alpha": 0, "zorder": 2},
        }
        final_results_data = {}
        logged_data = experiment_logger.to_dict()
        for key, val_list in logged_data.items():
            if len(val_list) > 0:
                final_results_data[key] = val_list[0]

        if final_results_data:
            plot_tracking_error_comparison(
                results_data=final_results_data,
                controller_map=controller_plot_map,
            )
        mse_tracking_experiment_plot(v_max, mse, traj_type_labels, model_vec, legends, [y_label], t_opt=t_opt, font_size=14)
