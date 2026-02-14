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
import time
import numpy as np
import matplotlib.pyplot as plt

from config.configuration_parameters import SimpleSimConfig
from config.gp_config import OnlineGPConfig
from config.paths import DEFAULT_MODEL_VERSION, DEFAULT_MODEL_NAME, DEFAULT_MODEL_TYPE
from src.core.controller import Quad3DMPC
from src.core.dynamics import Quadrotor3D
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.utils import load_pickled_models, interpol_mse, separate_variables
from src.utils.wind_model import RealisticWindModel
from src.utils.trajectories import random_trajectory, lemniscate_trajectory, loop_trajectory
from src.visualization.plotting import (
    initialize_drone_plotter, draw_drone_simulation, trajectory_tracking_results,
    get_experiment_files, mse_tracking_experiment_plot
)
from src.visualization.paper_plots import plot_tracking_error_comparison
from src.visualization.gp_online import visualize_gp_snapshot
from src.gp.rdrv import load_rdrv
from src.gp.utils import world_to_body_velocity_mapping
from src.gp.online import IncrementalGPManager
from src.utils.data_logger import DataLogger


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
         online_gp_pred_points=7):
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
    """

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
    quad_trajectory = np.zeros((len(reference_timestamps), len(quad_current_state)))
    u_optimized_seq = np.zeros((len(reference_timestamps), 4))

    # Sliding reference trajectory initial index
    current_idx = 0

    # Latency accounting:
    # 1) MPC optimize solve time only
    # 2) Online GP update/poll overhead
    # 3) Full control compute pipeline (excluding plant propagation/sleep)
    mpc_opt_time_acc = 0.0
    gp_update_time_acc = 0.0
    control_compute_time_acc = 0.0
    control_pre_sim_time_acc = 0.0
    control_post_sim_time_acc = 0.0

    # Measure total simulation time
    total_sim_time = 0.0

    # Initialize the online GP manager and history for main
    simulation_time = 0.0

    # --- Online GP Manager and History Lists Initialization (Moved outside the main while loop) ---
    history_gp_input_velocities = [] 
    history_gp_target_residuals = [] 
    history_timestamps_for_gp = []  # 记录时间戳
    
    out_online_gp_manager = None  # 用于快照可视化的在线GP管理器
    out_x_pred = None
    out_total_sim_time = 0.0
    out_snapshot_quality = None
    best_snapshot_score = -1.0
    visualized_all = False

    while (time.time() - start_time) < max_simulation_time and current_idx < reference_traj.shape[0]:
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
        
        # Check if we should use online GP predictions
        effective_query_velocities = None
        if online_gp_manager and use_online_gp and any(gp.is_trained_once for gp in online_gp_manager.gps):
            # 1. 使用上一步MPC规划的轨迹(x_pred)作为对未来状态的近似
            # 2. 将世界系速度转换为机体坐标系速度
            planned_states_body = world_to_body_velocity_mapping(x_pred)
            planned_velocities_body = planned_states_body[:, 7:10]
            n_plan = int(planned_velocities_body.shape[0])
            k_pred = int(np.clip(int(online_gp_pred_points), 1, n_plan))
            effective_query_velocities = planned_velocities_body[:k_pred, :]

            # 3. 在线GP仅预测前k个节点，尾部节点使用最后一个预测值进行平滑外推，
            #    以减少远期强外推对MPC的负面影响。
            predicted_residuals_k, predicted_variances_k = online_gp_manager.predict(effective_query_velocities)
            if k_pred < n_plan:
                online_predictions = np.tile(predicted_residuals_k[-1, :], (n_plan, 1))
                online_variances = np.tile(predicted_variances_k[-1, :], (n_plan, 1))
                online_predictions[:k_pred, :] = predicted_residuals_k
                online_variances[:k_pred, :] = predicted_variances_k
            else:
                online_predictions = predicted_residuals_k
                online_variances = predicted_variances_k
        # ========================================================================

        # Optimize control input to reach pre-set target
        t_opt_init = time.time()
        w_opt, x_pred = quad_mpc.optimize(use_model=model_ind, return_x=True, 
                                          online_gp_predictions=online_predictions,
                                          online_gp_variances=online_variances)
        mpc_opt_time_acc += time.time() - t_opt_init

        # Select first input (one for each motor) - MPC applies only first optimized input to the plant
        ref_u = np.squeeze(np.array(w_opt[:4]))

        # 可视化快照候选：选择“规划速度落在当前训练覆盖范围内比例更高”的时刻，
        # 避免展示明显超出学习区间的预测点。
        if (
            online_gp_manager and use_online_gp and x_pred is not None
            and any(gp.is_trained_once for gp in online_gp_manager.gps)
            and effective_query_velocities is not None
        ):
            try:
                future_velocities = np.array(effective_query_velocities, copy=False)
                dim_mix_scores = []
                dim_span_scores = []
                dim_in_ratios = []
                valid_dim_count = 0
                mixed_dim_count = 0
                for dim_idx, gp in enumerate(online_gp_manager.gps):
                    train_data = gp.buffer.get_training_set()
                    if len(train_data) < 8:
                        continue
                    vx = np.array([p[0] for p in train_data], dtype=float)
                    qv = future_velocities[:, dim_idx]
                    v_min, v_max = float(np.min(vx)), float(np.max(vx))
                    span = max(v_max - v_min, 1e-6)
                    margin = 0.10 * span
                    in_range = (qv >= (v_min - margin)) & (qv <= (v_max + margin))
                    in_ratio = float(np.mean(in_range))
                    valid_dim_count += 1
                    dim_in_ratios.append(in_ratio)
                    # 目标是“内插与外推并存”，希望比例在中间区间。
                    target_ratio = 0.60
                    mix_score = max(0.0, 1.0 - abs(in_ratio - target_ratio) / target_ratio)
                    if 0.20 <= in_ratio <= 0.85:
                        mixed_dim_count += 1
                    dim_mix_scores.append(mix_score)
                    dim_span_scores.append(float(np.max(qv) - np.min(qv)))

                if valid_dim_count > 0:
                    mixed_dim_ratio = mixed_dim_count / float(valid_dim_count)
                    score = (
                        2.0 * mixed_dim_ratio
                        + float(np.mean(dim_mix_scores))
                        + 0.15 * np.log1p(float(np.mean(dim_span_scores)))
                    )
                    if score >= best_snapshot_score:
                        best_snapshot_score = score
                        out_online_gp_manager = online_gp_manager
                        out_x_pred = np.array(x_pred, copy=True)
                        out_total_sim_time = float(total_sim_time)
                        out_snapshot_quality = {
                            "mixed_dim_ratio": float(mixed_dim_ratio),
                            "mean_in_ratio": float(np.mean(dim_in_ratios)),
                            "mean_pred_span": float(np.mean(dim_span_scores)),
                            "pred_points_used": int(future_velocities.shape[0]),
                        }
            except Exception:
                pass

        simulation_time = 0.0
        
        # --- ADDED: 在线GP的数据收集
        if online_gp_manager and use_online_gp: 
            s_before_sim  = quad_mpc.get_state()
            v_body_in = s_before_sim.T
            v_body_in = world_to_body_velocity_mapping(v_body_in)
            v_body_in = np.squeeze(v_body_in[:,7:10])  # Extract only the velocity components
            history_gp_input_velocities.append(v_body_in.copy())
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

            #存储数据
            history_gp_target_residuals.append(residual_acc_body.copy())
            history_timestamps_for_gp.append(total_sim_time) # 记录当前数据点的时间戳
            
            #更新在线GP
            update_start_time = time.time()

            # --- 异步更新与轮询 ---
            online_gp_manager.update(v_body_in, residual_acc_body, timestamp=total_sim_time)
            online_gp_manager.poll_for_results()

            gp_update_time_acc += time.time() - update_start_time
            #print(f"在线GP更新耗时: {time.time() - update_start_time:.4f}s")
        # --- END ADDED: Online GP Initialization ---
        post_sim_elapsed = time.perf_counter() - post_sim_start
        control_compute_time_acc += post_sim_elapsed
        control_post_sim_time_acc += post_sim_elapsed

        u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))
        current_idx += 1   

    quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
    quad_trajectory[-1, :] = np.expand_dims(quad_current_state, axis=0)
    u_optimized_seq[-1, :] = np.reshape(ref_u, (1, -1))
    
    # Average elapsed time per control step
    mean_opt_time = mpc_opt_time_acc / max(current_idx, 1)
    mean_gp_update_time = gp_update_time_acc / max(current_idx, 1)
    mean_total_ctrl_time = control_compute_time_acc / max(current_idx, 1)
    mean_pre_sim_ctrl_time = control_pre_sim_time_acc / max(current_idx, 1)
    mean_post_sim_ctrl_time = control_post_sim_time_acc / max(current_idx, 1)

    rmse = interpol_mse(reference_timestamps, reference_traj[:, :3], reference_timestamps, quad_trajectory[:, :3])
    max_vel = np.max(np.sqrt(np.sum(reference_traj[:, 7:10] ** 2, 1)))

    # Use model_label for the title
    title = rf'${model_label}: \, v_{{\mathrm{{max}}}} = {max_vel:.2f} \, \mathrm{{m/s}}, \, RMSE = {rmse:.3f} \, \mathrm{{m}}$'
    
    print(f'\n--- Simulation finished ---\n')
    print(f'Average optimization time (solve only): {mean_opt_time:.4f} s')
    print(f'Average online update overhead: {mean_gp_update_time:.4f} s')
    print(f'Average control compute time (full pipeline, no sleep/sim): {mean_total_ctrl_time:.4f} s')
    print(f'  - pre-sim compute: {mean_pre_sim_ctrl_time:.4f} s')
    print(f'  - post-sim compute: {mean_post_sim_ctrl_time:.4f} s')
    print(f'RMSE: {rmse:.4f} m')
    print(f'Maximum velocity: {max_vel:.2f} m/s')

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
                max_pred_points=int(online_gp_pred_points),
            )
            #out_online_gp_manager.visualize_training_history()
        
        # trajectory_tracking_results(reference_timestamps, reference_traj, quad_trajectory,
        #                             reference_u, u_optimized_seq, title)

        # tracking_results_with_wind(
        #     t_ref=reference_timestamps,
        #     x_ref=reference_traj,
        #     x_executed=quad_trajectory,
        #     title=title,
        #     wind_model=wind_model # <-- 关键改动
        # )

    # --- 新增：绘制在线GP结果 ---
    if online_gp_manager and history_gp_input_velocities and history_gp_target_residuals and visualized_all:
        print("\n--- Plotting Online GP Collected Data: Input Velocity vs. Target Residual ---")
        history_gp_input_velocities_arr = np.array(history_gp_input_velocities)
        history_gp_target_residuals_arr = np.array(history_gp_target_residuals)

        num_dims_to_plot = online_gp_manager.num_dimensions
        # dim_labels = ['机体Vx', '机体Vy', '机体Vz'] 
        dim_labels = ['Body Vx', 'Body Vy', 'Body Vz'] # 使用英文

        fig_scatter, axes_scatter = plt.subplots(num_dims_to_plot, 1, figsize=(3.5, 2 * num_dims_to_plot), sharex=False, sharey=False)
        if num_dims_to_plot == 1: axes_scatter = [axes_scatter]

        for i in range(num_dims_to_plot):
            ax = axes_scatter[i]
            if history_gp_input_velocities_arr.shape[0] > 0 and history_gp_target_residuals_arr.shape[0] > 0:
                 ax.scatter(history_gp_input_velocities_arr[:, i], 
                            history_gp_target_residuals_arr[:, i], 
                            # label=f'收集的数据 ({dim_labels[i]})', 
                            label=f'Collected Data ({dim_labels[i]})', # 使用英文
                            alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
            # ax.set_xlabel(f'输入机体速度分量 {dim_labels[i]} (m/s)')
            ax.set_xlabel(f'Input Body Velocity {dim_labels[i]} (m/s)') # 使用英文
            # ax.set_ylabel('目标加速度残差分量 (m/s^2)')
            ax.set_ylabel('Target Acceleration Residual (m/s^2)') # 使用英文
            ax.legend()
            ax.grid(True)
        
        plt.savefig("online_gp_collected_data.pdf", bbox_inches="tight")
        if SimpleSimConfig.show_intermediate_plots:
            plt.show()
        else:
            plt.close()
    # --- 在线GP绘图结束 ---
    # --- 修改 1: 增加函数返回值，用于后续保存 ---
    return (
        rmse,
        max_vel,
        mean_opt_time,
        reference_timestamps,
        reference_traj,
        quad_trajectory,
        mean_gp_update_time,
        mean_total_ctrl_time,
    )
    # --- 修改结束 ---


if __name__ == '__main__':
    # --- 核心修复: 添加CUDA安全的多进程启动方法 ---
    # 必须在任何其他多进程或CUDA操作之前调用
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # 如果上下文已经设置，可能会抛出此异常，属于正常情况
        pass

    # --- 修复结束 ---
    # Trajectory options
    traj_type_vec = [{"random": 1}]
    traj_type_labels = ["Random"]
    
    # av_speed_vec = [[1.5,2.0,2.5,3.0,3.5],
    #                 [12.0],
    #                 [12.0]]
    av_speed_vec = [[3.0],
                    [12.0],
                    [12.0]]
    # traj_type_vec = [{"random": 1}, "loop", "lemniscate"]
    # traj_type_labels = ["Random", "Circle", "Lemniscate"]

    # av_speed_vec = [[2.0, 3.5],
    #                 [2.0, 12.0],
    #                 [2.0, 12.0]]

    # av_speed_vec = [[2.0, 2.7, 3.0, 3.2, 3.5],
    #                 [2.0, 4.5, 7.0, 9.5, 12.0],
    #                 [2.0, 4.5, 7.0, 9.5, 12.0]]
    git_list = DEFAULT_MODEL_VERSION
    name_list = DEFAULT_MODEL_NAME
    type_list = DEFAULT_MODEL_TYPE

    # Simulation options
    plot_sim = SimpleSimConfig.custom_sim_gui
    noisy_sim_options = SimpleSimConfig.simulation_disturbances
    
    # 只运行AR-MPC模型（在线GP增量修正）
    model_vec = [{
        "simulation_options": noisy_sim_options,
        "model": {"version": git_list, "name": name_list, "reg_type": type_list, 'use_online_gp': True},
        "description": "AR-MPC" # Added description
    }]
    legends = ['AR']
    
    # 初始化数据记录器
    experiment_logger = DataLogger()
    
    y_label = "RMSE [m]"
    # Define result vectors
    mse = np.zeros((len(traj_type_vec), len(av_speed_vec[0]), len(model_vec)))
    v_max = np.zeros((len(traj_type_vec), len(av_speed_vec[0])))
    t_opt = np.zeros((len(traj_type_vec), len(av_speed_vec[0]), len(model_vec)))

    for n_train_id, model_type in enumerate(model_vec):
        # Initialize online GP manager if needed
        online_gp_manager = None
        use_online_gp = False
        if model_type["model"] and model_type["model"].get("use_online_gp", False):
            use_online_gp = True
            online_gp_manager = IncrementalGPManager(config=OnlineGPConfig().to_dict())
        
        if model_type["model"] is not None:
            custom_mpc = prepare_quadrotor_mpc(model_type["simulation_options"], **model_type["model"])
            use_offline_gp = True
        else:
            custom_mpc = prepare_quadrotor_mpc(model_type["simulation_options"])
            use_offline_gp = False

        model_desc = model_type.get("description", legends[n_train_id])

        for traj_id, traj_type in enumerate(traj_type_vec):

            for v_id, speed in enumerate(av_speed_vec[traj_id]):

                traj_params = {"av_speed": speed, "reference_type": traj_type, "plot": plot_sim}

                # --- 核心修改：在每次新的速度测试开始时，重置GP管理器的状态 ---
                if online_gp_manager:
                    online_gp_manager.reset()
                # --- 修改结束 --
                run_result = main(
                    custom_mpc,
                    **traj_params,
                    use_online_gp=use_online_gp,
                    use_offline_gp=use_offline_gp,
                    model_label=model_desc,
                    online_gp_manager=online_gp_manager,
                )
                (
                    mse[traj_id, v_id, n_train_id],
                    traj_v,
                    opt_dt,
                    t_ref,
                    x_ref,
                    x_executed,
                    mean_gp_update_dt,
                    mean_ctrl_dt,
                ) = run_result
                
                # 统一记录“完整控制计算时间”（不含sleep/仿真推进）。
                t_opt[traj_id, v_id, n_train_id] += mean_ctrl_dt
                if v_max[traj_id, v_id] == 0:
                    v_max[traj_id, v_id] = traj_v

                # --- 修改 4: 将当前运行的结果存储在内存变量中 ---
                # 我们只保存最后一次速度测试的结果作为绘图代表
                if v_id == len(av_speed_vec[traj_id]) - 1:
                    controller_name = legends[n_train_id]
                    # 使用 DataLogger 存储
                    result_data = {
                        't_ref': t_ref,
                        'x_ref': x_ref,
                        'x_executed': x_executed,
                    }
                    experiment_logger.log(controller_name, result_data)
                # --- 修改结束 ---
        # --- 核心修改 3: 在模型的所有速度测试结束后，再关闭管理器 ---
        if online_gp_manager:
            online_gp_manager.shutdown()

    _, err_file, v_file, t_file = get_experiment_files()
    np.save(err_file, mse)
    np.save(v_file, v_max)
    np.save(t_file, t_opt)
    
    # --- 修改 6: 在所有实验结束后，调用新的对比绘图函数 ---
    # 生成绘图（静默模式）

    # plot_combined_results(combined_plot_data)

    # 定义每个控制器的绘图样式
    controller_plot_map = {
        'AR': {'color': '#7C7CBA', 'linestyle': '-', 'linewidth': 1, 'label': 'AR-MPC', 'fill_alpha': 0.25, 'zorder': 4},
        'SGP': {'color': '#EECA40', 'linestyle': '-', 'linewidth': 1, 'label': 'SGP-MPC', 'fill_alpha': 0.05, 'zorder': 3},
        'nominal': {'color': '#23BAC5', 'linestyle': '-', 'linewidth': 1, 'label': 'Nominal MPC', 'fill_alpha': 0.15, 'zorder': 1},
        'perfect': {'color': '#2ecc71', 'linestyle': '--', 'linewidth': 1, 'label': 'Perfect Model', 'fill_alpha': 0,'zorder': 2},
    }
    
    # 调用新的绘图函数，传入内存中的数据字典
    # DataLogger.data 存储的是 list，我们需要的是 {controller_name: result_data}
    # 因为我们每个控制器只log了一次result_data，所以这里取 experiment_logger.data[key][0]
    
    final_results_data = {}
    logged_data = experiment_logger.to_dict()
    for key, val_list in logged_data.items():
        if len(val_list) > 0:
            final_results_data[key] = val_list[0]
            
    plot_tracking_error_comparison(
        results_data=final_results_data,
        controller_map=controller_plot_map
    )
    # --- 修改结束 ---
    
    mse_tracking_experiment_plot(v_max, mse, traj_type_labels, model_vec, legends, [y_label], t_opt=t_opt, font_size=14)

# python src/experiments/comparative_experiment.py --model_version 89954f3 --model_name simple_sim_gp --model_type gp --fast
