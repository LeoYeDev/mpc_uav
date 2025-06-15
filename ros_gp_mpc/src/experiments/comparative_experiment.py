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
import argparse
import numpy as np
import torch 

from config.configuration_parameters import SimpleSimConfig
from src.quad_mpc.quad_3d_mpc import Quad3DMPC
from src.quad_mpc.quad_3d import Quadrotor3D
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.utils import load_pickled_models, interpol_mse, separate_variables
from src.utils.visualization import initialize_drone_plotter, draw_drone_simulation, trajectory_tracking_results, \
    get_experiment_files
from src.utils.visualization import mse_tracking_experiment_plot
from src.utils.trajectories import random_trajectory, lemniscate_trajectory, loop_trajectory
from src.model_fitting.rdrv_fitting import load_rdrv
from src.model_fitting.gp_common import world_to_body_velocity_mapping

global model_num

######
from src.model_fitting.gp_online import *
import matplotlib.pyplot as plt
from src.model_fitting.gp_online import IncrementalGPManager
from src.model_fitting.gp_online_visualization import visualize_gp_snapshot
from src.model_fitting.gp_common import world_to_body_velocity_mapping 
######
def prepare_quadrotor_mpc(simulation_options, version=None, name=None, reg_type="gp", quad_name=None,
                          t_horizon=1.0, q_diagonal=None, r_diagonal=None, q_mask=None):
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
        quad_name = "my_quad_" + str(globals()['model_num'])
        globals()['model_num'] += 1

    # Initialize quad MPC
    quad_mpc = Quad3DMPC(my_quad, t_horizon=t_horizon, optimization_dt=node_dt, simulation_dt=simulation_dt,
                         q_cost=q_diagonal, r_cost=r_diagonal, n_nodes=n_mpc_nodes,
                         pre_trained_models=pre_trained_models, model_name=quad_name, q_mask=q_mask, rdrv_d_mat=rdrv_d)

    return quad_mpc


def main(quad_mpc, av_speed, reference_type=None, plot=False,use_gp_ject=False):
    """

    :param quad_mpc:
    :type quad_mpc: Quad3DMPC
    :param av_speed:
    :param reference_type:
    :param plot:
    :return:
    """

    # Recover some necessary variables from the MPC object
    my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    simulation_dt = quad_mpc.simulation_dt
    t_horizon = quad_mpc.t_horizon

    reference_over_sampling = 5
    mpc_period = t_horizon / (n_mpc_nodes * reference_over_sampling)
    #预测时长1s，一周期内点数为10，启用mpc周期0.02s

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
            quad=my_quad, discretization_dt=mpc_period, seed=reference_type["random"], speed=av_speed, plot=plot)

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
    max_simulation_time = 10000

    ref_u = reference_u[0, :]
    quad_trajectory = np.zeros((len(reference_timestamps), len(quad_current_state)))
    u_optimized_seq = np.zeros((len(reference_timestamps), 4))

    # Sliding reference trajectory initial index
    current_idx = 0

    # Measure the MPC optimization time
    mean_opt_time = 0.0

    # Measure total simulation time
    total_sim_time = 0.0

    # Initialize the online GP manager and history for main
    simulation_time = 0.0

    # --- Online GP Manager and History Lists Initialization (Moved outside the main while loop) ---
    snapshot_visualization_done = False
    history_gp_input_velocities = [] 
    history_gp_target_residuals = [] 
    history_timestamps_for_gp = []  # 记录时间戳
    collect_online_gp_data_flag = True  # Set to True to enable online GP data collection
    # use_gp_ject = True
    online_gp_manager = None  # Initialize as None, will be set if collect_online_gp_data_flag is True
    visualized_all = False

    if collect_online_gp_data_flag and use_gp_ject: 
        print("\n" + "="*50)
        print("在线GP模块已激活：正在初始化并进行预热...")
        print("="*50)
        # 使用我们最终确定的、更稳健的配置
        online_gp_config = {
            'num_dimensions': 3,
            'main_process_device': 'cpu',
            'worker_device_str': 'cpu',
            'buffer_level_capacities': [10, 20, 40], # 三层缓冲区容量
            'buffer_level_sparsity': [1, 2, 5],      # 稀疏因子：每1/2/5个点存入
            'min_points_for_initial_train': 30,      # 触发首次训练的最小数据点
            'min_points_for_ema': 30,                # 启用EMA所需的最小数据点
            'refit_hyperparams_interval': 30,       # 触发再训练的更新次数间隔
            'worker_train_iters': 40,               # 后台训练迭代次数
            'worker_lr': 0.04,                       # 训练学习率
            'ema_alpha': 0.05,                       # EMA平滑系数
        }
        online_gp_manager = IncrementalGPManager(config=online_gp_config)

    while (time.time() - start_time) < max_simulation_time and current_idx < reference_traj.shape[0]:

        quad_current_state = my_quad.get_state(quaternion=True, stacked=True)

        quad_trajectory[current_idx, :] = np.expand_dims(quad_current_state, axis=0)
        u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

        # ##### Optimization runtime (outer loop) ##### #
        # Get the chunk of trajectory required for the current optimization
        ref_traj_chunk, ref_u_chunk = get_reference_chunk(
            reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling)

        model_ind = quad_mpc.set_reference(x_reference=separate_variables(ref_traj_chunk), u_reference=ref_u_chunk)

        # Optimize control input to reach pre-set target
        t_opt_init = time.time()
        w_opt, x_pred = quad_mpc.optimize(use_model=model_ind, return_x=True)
        mean_opt_time += time.time() - t_opt_init

        # Select first input (one for each motor) - MPC applies only first optimized input to the plant
        ref_u = np.squeeze(np.array(w_opt[:4]))

        if len(quad_trajectory) > 0 and plot and current_idx > 0:
            draw_drone_simulation(real_time_artists, quad_trajectory[:current_idx, :], my_quad, targets=None,
                                  targets_reached=None, pred_traj=x_pred, x_pred_cov=None)

        simulation_time = 0.0
        
        # --- ADDED 
        if collect_online_gp_data_flag and use_gp_ject : 
            s_before_sim  = quad_mpc.get_state()
            v_body_in = s_before_sim .T
            v_body_in = world_to_body_velocity_mapping(v_body_in)
            v_body_in = np.squeeze(v_body_in[:,7:10])  # Extract only the velocity components
            history_gp_input_velocities.append(v_body_in.copy())
        # --- END ADDED:

        # ##### Simulation runtime (inner loop) ##### #
        while simulation_time < mpc_period:
            simulation_time += simulation_dt
            total_sim_time += simulation_dt
            quad_mpc.simulate(ref_u)


        # --- ADDED: Online GP Initialization and History ---
        if online_gp_manager : 
            #推演后的状态
            s_after_sim  = quad_mpc.get_state()
            v_body_out = s_after_sim .T
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

            # 实际加速度偏差
            # x_predic2, _ = quad_mpc.forward_prop(np.squeeze(quad_current_state), w_opt=w_opt[:4],
            #                                   t_horizon=simulation_time, use_gp=False)
            # x_predic2 = x_predic2[[-1], :]
            # x_predic2 = world_to_body_velocity_mapping(x_predic2)
            # x_predic2 = np.squeeze(x_predic2[:,7:10])  # Extract only the velocity components
            # print(f"------: GP Predicted state: {x_predic}, Model predicted state: {x_predic2}")

            residual_acc_body = v_body_out - v_body_predic
            residual_acc_body /= simulation_time

            #存储数据
            history_gp_target_residuals.append(residual_acc_body.copy())
            history_timestamps_for_gp.append(total_sim_time) # 记录当前数据点的时间戳
            # print(f"*******: Collected data for online GP: {x_in} -> {y_err}")
            
            #更新在线GP
            update_start_time = time.time()

            # --- 异步更新与轮询 ---
            online_gp_manager.update(v_body_in, residual_acc_body)
            online_gp_manager.poll_for_results()

            mean_opt_time += time.time() - update_start_time
            print(f"在线GP更新耗时: {time.time() - update_start_time:.4f}s")

            # --- 在初始优化后可视化GP拟合情况 (一次) ---
            if total_sim_time > 5.0 and not snapshot_visualization_done:
                # 检查是否有任何一个GP维度已经训练过了
                if any(gp.is_trained_once for gp in online_gp_manager.gps):
                    print(f"\n📸 [快照] 仿真时间 {total_sim_time:.2f}s, 生成当前GP回归效果快照...")
                    visualize_gp_snapshot(
                        online_gp_manager=online_gp_manager,
                        # 使用当前的MPC预测轨迹来定义绘图的X轴范围
                        mpc_planned_states=x_pred, 
                        snapshot_info_str=f"In-Flight Snapshot @ SimTime {total_sim_time:.2f}s"
                    )
                online_gp_manager.visualize_training_history()
                snapshot_visualization_done = True
            # --- 初始优化后可视化结束 ---
        # --- END ADDED: Online GP Initialization ---


        u_optimized_seq[current_idx, :] = np.reshape(ref_u, (1, -1))
        current_idx += 1   

    if online_gp_manager:
            online_gp_manager.shutdown()

    quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
    quad_trajectory[-1, :] = np.expand_dims(quad_current_state, axis=0)
    u_optimized_seq[-1, :] = np.reshape(ref_u, (1, -1))
    
    # Average elapsed time per optimization
    mean_opt_time /= current_idx

    rmse = interpol_mse(reference_timestamps, reference_traj[:, :3], reference_timestamps, quad_trajectory[:, :3])
    max_vel = np.max(np.sqrt(np.sum(reference_traj[:, 7:10] ** 2, 1)))

    with_gp = ' + GP ' if quad_mpc.gp_ensemble is not None else ' - GP '
    title = r'$v_{max}$=%.2f m/s | RMSE: %.4f m | %s ' % (max_vel, float(rmse), with_gp)

    print(f'\n--- Simulation finished ---\n')
    print(f'Average optimization time: {mean_opt_time:.4f} s')
    print(f'RMSE: {rmse:.4f} m')
    print(f'Maximum velocity: {max_vel:.2f} m/s')

    if plot:
        trajectory_tracking_results(reference_timestamps, reference_traj, quad_trajectory,
                                    reference_u, u_optimized_seq, title)

    # --- 新增：绘制在线GP结果 ---
    if online_gp_manager and history_gp_input_velocities and history_gp_target_residuals and visualized_all:
        print("\n--- Plotting Online GP Collected Data: Input Velocity vs. Target Residual ---")
        history_gp_input_velocities_arr = np.array(history_gp_input_velocities)
        history_gp_target_residuals_arr = np.array(history_gp_target_residuals)

        num_dims_to_plot = online_gp_manager.num_dimensions
        # dim_labels = ['机体Vx', '机体Vy', '机体Vz'] 
        dim_labels = ['Body Vx', 'Body Vy', 'Body Vz'] # 使用英文

        fig_scatter, axes_scatter = plt.subplots(num_dims_to_plot, 1, figsize=(10, 5 * num_dims_to_plot), sharex=False, sharey=False)
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
            # ax.set_title(f'在线GP数据: 输入 {dim_labels[i]} vs. 目标残差 {dim_labels[i]}')
            ax.set_title(f'Online GP Data: Input {dim_labels[i]} vs. Target Residual {dim_labels[i]}') # 使用英文
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
        
        # fig_scatter.suptitle(f'在线GP收集数据：速度-残差关系\n(实验: {quad_mpc.model_name}, 轨迹: {str(reference_type)}, 速度: {av_speed:.1f} m/s)', fontsize=14)
        fig_scatter.suptitle(f'Online GP Collected Data: Velocity-Residual Relationship\n(Traj: {str(reference_type)}, Speed: {av_speed:.1f} m/s)', fontsize=14) # 使用英文
        fig_scatter.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show() # 将show调用移到main函数末尾，以显示所有图
    # --- 在线GP绘图结束 ---
    return rmse, max_vel, mean_opt_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_version", type=str, default="", nargs="+",
                        help="Versions to load for the regression models. By default it is an 8 digit git hash."
                             "Must specify the version for each model separated by spaces.")

    parser.add_argument("--model_name", type=str, default="", nargs="+",
                        help="Name of the regression models within the specified <model_version> folders. "
                             "Must specify the names for all models separated by spaces.")

    parser.add_argument("--model_type", type=str, default="", nargs="+",
                        help="Type of regression models (GP or RDRv linear). "
                             "Must be specified for all models separated by spaces.")

    parser.add_argument("--fast", dest="fast", action="store_true",
                        help="Set to True to run a fast experiment with less velocity samples.")
    parser.set_defaults(fast=False)

    input_arguments = parser.parse_args()

    globals()['model_num'] = 0

    # Trajectory options
    # traj_type_vec = [{"random": 1}, "loop", "lemniscate"]
    # traj_type_labels = ["Random", "Circle", "Lemniscate"]

    # if input_arguments.fast:
    #     av_speed_vec = [[2.0, 3.5],
    #                     [2.0, 12.0],
    #                     [2.0, 12.0]]
    # else:
    #     av_speed_vec = [[2.0, 2.7, 3.0, 3.2, 3.5],
    #                     [2.0, 4.5, 7.0, 9.5, 12.0],
    #                     [2.0, 4.5, 7.0, 9.5, 12.0]]

    traj_type_vec = [{"random": 1}]
    traj_type_labels = ["Random"]
    
    if input_arguments.fast:
        av_speed_vec = [[3.5],
                        [12.0],
                        [12.0]]

    # Model options
    git_list = input_arguments.model_version
    name_list = input_arguments.model_name
    type_list = input_arguments.model_type

    assert len(git_list) == len(name_list) == len(type_list)

    # Simulation options
    plot_sim = SimpleSimConfig.custom_sim_gui
    noisy_sim_options = SimpleSimConfig.simulation_disturbances
    perfect_sim_options = {"payload": False, "drag": False, "noisy": False, "motor_noise": False}
    model_vec = [
        {"simulation_options": perfect_sim_options, "model": None},
        {"simulation_options": noisy_sim_options, "model": None}]

    legends = ['perfect', 'nominal']
    for git, m_name, gp_or_rdrv in zip(git_list, name_list, type_list):
        model_vec += [{"simulation_options": noisy_sim_options,
                       "model": {"version": git, "name": m_name, "reg_type": gp_or_rdrv}}]
        legends += [gp_or_rdrv + ": " + m_name]

    y_label = "RMSE [m]"

    # Define result vectors
    mse = np.zeros((len(traj_type_vec), len(av_speed_vec[0]), len(model_vec)))
    v_max = np.zeros((len(traj_type_vec), len(av_speed_vec[0])))
    t_opt = np.zeros((len(traj_type_vec), len(av_speed_vec[0]), len(model_vec)))

    for n_train_id, model_type in enumerate(model_vec):

        if model_type["model"] is not None:
            custom_mpc = prepare_quadrotor_mpc(model_type["simulation_options"], **model_type["model"])
            use_gp_ject = True
        else:
            custom_mpc = prepare_quadrotor_mpc(model_type["simulation_options"])
            use_gp_ject = False

        for traj_id, traj_type in enumerate(traj_type_vec):

            for v_id, speed in enumerate(av_speed_vec[traj_id]):

                traj_params = {"av_speed": speed, "reference_type": traj_type, "plot": plot_sim}

                mse[traj_id, v_id, n_train_id], traj_v, opt_dt = main(custom_mpc, **traj_params, use_gp_ject=use_gp_ject)
                t_opt[traj_id, v_id, n_train_id] += opt_dt

                if v_max[traj_id, v_id] == 0:
                    v_max[traj_id, v_id] = traj_v

    _, err_file, v_file, t_file = get_experiment_files()
    np.save(err_file, mse)
    np.save(v_file, v_max)
    np.save(t_file, t_opt)

    # from src.utils.visualization import load_past_experiments
    # _, mse, v_max, t_opt = load_past_experiments()

    mse_tracking_experiment_plot(v_max, mse, traj_type_labels, model_vec, legends, [y_label], t_opt=t_opt, font_size=26)

# python src/experiments/comparative_experiment.py --model_version 89954f3 --model_name simple_sim_gp --model_type gp --fast