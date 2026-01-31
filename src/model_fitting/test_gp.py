import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

# Assuming gp_online.py is in src.model_fitting relative to where this script is run
# or PYTHONPATH is set up correctly.
from src.model_fitting.gp_online import IncrementalGPManager 
# If ExactIncrementalGPModel and MultiLevelBuffer are also in gp_online.py and imported with *,
# this single import should suffice.

class SimplifiedQuadModel:
    """A simplified drone model for testing the online GP."""
    def __init__(self, mass=1.0, disturbance_func=None, is_true_model=False):
        self.mass = mass
        self.state = np.zeros(12) # [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        self.dt = 0.02
        self.disturbance_func = disturbance_func
        self.is_true_model = is_true_model

    def ode(self, state, control_input):
        g = 9.81
        ax_nom = control_input[0] / self.mass
        ay_nom = control_input[1] / self.mass
        az_nom = control_input[2] / self.mass - g
        acc_nominal = np.array([ax_nom, ay_nom, az_nom])
        
        x_dot = np.zeros_like(state)
        x_dot[0:3] = state[6:9]
        x_dot[6:9] = acc_nominal
        return x_dot

    def step(self, control_input, current_time_step_for_disturbance=0):
        state_derivative_nominal = self.ode(self.state, control_input)
        acc_nominal = state_derivative_nominal[6:9]

        actual_total_acceleration = np.copy(acc_nominal)
        if self.is_true_model and self.disturbance_func:
            external_disturbance_acc = self.disturbance_func(
                current_time_step_for_disturbance, self.state, control_input
            )
            actual_total_acceleration += external_disturbance_acc
        
        self.state[6:9] += actual_total_acceleration * self.dt
        self.state[0:3] += self.state[6:9] * self.dt
        return actual_total_acceleration

    def get_state(self):
        return np.copy(self.state)

    def get_velocity(self):
        return np.copy(self.state[6:9])

def main_test_online_gp():
    print("开始在线GP独立测试仿真 (v5)...")

    sim_steps = 500
    dt = 0.02
    num_dimensions_gp = 3 # Consistent with typical drone dynamics

    def true_world_disturbance(t_idx, state, control):
        disturbance = np.zeros(3)
        disturbance[0] = 0.8 * np.sin(t_idx * dt * 0.6 + 0.1)
        if 120 < t_idx < 220:
            disturbance[1] = 1.2 * np.cos((t_idx - 120) * dt * 1.5)
        disturbance[2] = -0.25 + 0.1 * np.sin(t_idx * dt * 0.3)
        return disturbance

    true_quad = SimplifiedQuadModel(mass=1.0, disturbance_func=true_world_disturbance, is_true_model=True)
    true_quad.dt = dt
    nominal_quad = SimplifiedQuadModel(mass=1.0)
    nominal_quad.dt = dt

    def long_term_gp_pred_func(velocity_array): # velocity_array is (3,)
        # Example: return np.array([0.05 * np.sin(velocity_array[0]), -0.03, 0.02])
        return np.zeros(3)

    # Parameters for IncrementalGPManager, matching the user's __init__ signature
    gp_params_manager = {
        "num_dimensions": num_dimensions_gp,
        "min_points_for_initial_train": 30,
        "device_str": 'cpu',
        "ema_alpha": 0.05,
        "buffer_level_capacities": [60, 40, 30], 
        "buffer_level_sparsity": [1, 2, 4],    
        "initial_lr": 0.03,
        "periodic_refit_lr": 0.01,
        "online_finetune_lr": 0.001,
        "initial_train_iters": 100,
        "periodic_refit_iters": 20,
        "online_finetune_iters": 0, 
        "patience_epochs_initial": 10,
        "early_stopping_tol_initial": 1e-3,
        "refit_hyperparams_interval": 50
    }
    
    online_gp_manager = IncrementalGPManager(**gp_params_manager)
    print("在线GP管理器已初始化。")

    # History storage
    history_velocities_np = np.zeros((sim_steps, num_dimensions_gp))
    history_residuals_for_training_np = np.zeros((sim_steps, num_dimensions_gp))
    history_online_gp_preds_mean_np = np.zeros((sim_steps, num_dimensions_gp))
    history_online_gp_preds_var_np = np.zeros((sim_steps, num_dimensions_gp))
    history_actual_accelerations_np = np.zeros((sim_steps, num_dimensions_gp))
    history_augmented_model_accelerations_np = np.zeros((sim_steps, num_dimensions_gp))

    base_thrust_z = nominal_quad.mass * 9.81
    control_inputs_sequence = [np.array([0.0, 0.0, base_thrust_z])] * sim_steps
    for i in range(50, 100): control_inputs_sequence[i] = np.array([1.5, 0.0, base_thrust_z])
    for i in range(150, 200): control_inputs_sequence[i] = np.array([-1.5, 0.0, base_thrust_z])
    for i in range(250, 300): control_inputs_sequence[i] = np.array([0.0, 1.0, base_thrust_z + 2.0])
    for i in range(350, 400): control_inputs_sequence[i] = np.array([0.5, -0.5, base_thrust_z - 1.0])

    print(f"开始 {sim_steps} 步仿真循环...")
    for i in range(sim_steps):
        current_control = control_inputs_sequence[i]
        current_true_state = true_quad.get_state()
        current_velocity_np = current_true_state[6:9] 

        actual_acceleration_np = true_quad.step(current_control, current_time_step_for_disturbance=i)
        
        nominal_state_derivative = nominal_quad.ode(current_true_state, current_control)
        nominal_model_acceleration_np = nominal_state_derivative[6:9]

        long_term_gp_pred_np = long_term_gp_pred_func(current_velocity_np)
        augmented_model_acceleration_np = nominal_model_acceleration_np + long_term_gp_pred_np
        residual_for_online_gp_training_np = actual_acceleration_np - augmented_model_acceleration_np
        
        current_velocities_list = [current_velocity_np[d].item() for d in range(num_dimensions_gp)]
        residuals_for_update_list = [residual_for_online_gp_training_np[d].item() for d in range(num_dimensions_gp)]

        update_start_time = time.time()
        online_gp_manager.update(current_velocities_list, residuals_for_update_list)
        update_duration = time.time() - update_start_time

        query_velocities_for_predict = [[current_velocity_np[d].item()] for d in range(num_dimensions_gp)]
        pred_mean_list, pred_var_list = online_gp_manager.predict(query_velocities_for_predict)

        history_velocities_np[i, :] = current_velocity_np
        history_residuals_for_training_np[i, :] = residual_for_online_gp_training_np
        history_online_gp_preds_mean_np[i, :] = [m[0] for m in pred_mean_list] 
        history_online_gp_preds_var_np[i, :] = [v[0] for v in pred_var_list]   
        history_actual_accelerations_np[i, :] = actual_acceleration_np
        history_augmented_model_accelerations_np[i, :] = augmented_model_acceleration_np

        if i % 50 == 0 or i == sim_steps - 1:
            print(f"Sim step {i+1}/{sim_steps}, OnlineGP Update Time: {update_duration*1000:.2f} ms")
            print(f"  Current Vel: {np.round(current_velocity_np,3)}")
            print(f"  Actual Acc: {np.round(actual_acceleration_np,3)}")
            print(f"  Residual for Training: {np.round(residual_for_online_gp_training_np,3)}")
            print(f"  OnlineGP Pred Mean: {np.round(np.array([m[0] for m in pred_mean_list]),3)}")

    print("仿真循环结束。")
    print("生成可视化图像...")
    
    dims = ['X', 'Y', 'Z']
    time_axis = np.arange(sim_steps) * dt

    for dim_idx in range(num_dimensions_gp):
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, history_velocities_np[:, dim_idx], label=f'Velocity {dims[dim_idx]} (Input to GP)')
        plt.title(f'Dimension {dims[dim_idx]} - Online GP Test (v5)')
        plt.ylabel('Velocity (m/s)')
        plt.legend(); plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time_axis, history_actual_accelerations_np[:, dim_idx], label=f'Actual Accel {dims[dim_idx]}', alpha=0.7)
        plt.plot(time_axis, history_augmented_model_accelerations_np[:, dim_idx], label=f'Augmented Model Accel {dims[dim_idx]}', linestyle=':', alpha=0.7)
        plt.plot(time_axis, history_residuals_for_training_np[:, dim_idx], label=f'True Residual {dims[dim_idx]} (Target)', color='red', linestyle='--')
        plt.plot(time_axis, history_online_gp_preds_mean_np[:, dim_idx], label=f'OnlineGP Pred Mean {dims[dim_idx]}', color='green', linestyle='-.')
        plt.ylabel('Acceleration / Residual (m/s^2)')
        plt.legend(); plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time_axis, history_residuals_for_training_np[:, dim_idx], label=f'True Residual {dims[dim_idx]} (Target)', color='red', linestyle='-')
        plt.plot(time_axis, history_online_gp_preds_mean_np[:, dim_idx], label=f'OnlineGP Predicted Mean {dims[dim_idx]}', color='green', linestyle='--')
        
        std_dev = np.sqrt(np.maximum(history_online_gp_preds_var_np[:, dim_idx], 1e-9))
        lower_bound = history_online_gp_preds_mean_np[:, dim_idx] - 1.96 * std_dev
        upper_bound = history_online_gp_preds_mean_np[:, dim_idx] + 1.96 * std_dev
        plt.fill_between(time_axis, lower_bound, upper_bound, color='green', alpha=0.2, label='OnlineGP 95% CI')
        
        plt.xlabel('Time (s)'); plt.ylabel('Residual Value (m/s^2)')
        plt.legend(); plt.grid(True)
        
        # plt.tight_layout()
        # file_name = f"online_gp_test_v5_dim_{dims[dim_idx]}.png"
        # plt.savefig(file_name)
        # print(f"图像 {file_name} 已保存。")

    # Internal visualization call
    if online_gp_manager.models and len(online_gp_manager.models) > 0 and online_gp_manager.models[0] is not None:
        first_model_can_visualize = False
        # Check if the model is ready for visualization (e.g., has been trained at least once)
        # A simple check could be if train_x_normalized for that dimension is not None
        if online_gp_manager.train_x_normalized and \
           len(online_gp_manager.train_x_normalized) > dim_idx and \
           online_gp_manager.train_x_normalized[0] is not None:
            first_model_can_visualize = True
        
        if first_model_can_visualize:
            print("调用 IncrementalGPManager 的内部可视化函数 (维度0)...")
            # Call visualize_single_dimension for its side effect (plotting)
            # It does not return fig, axes based on the provided gp_online.py
            online_gp_manager.visualize_single_dimension(dim_idx=0) 
            
            # Attempt to get the current figure (created by visualize_single_dimension) and save it
            # try:
            #     if plt.get_fignums(): # Check if there are any figures
            #         fig_to_save = plt.figure(plt.get_fignums()[-1]) # Get the last created figure
            #         fig_to_save.suptitle("Internal GP Visualization - Dimension X (from Manager v5)")
            #         plt.savefig("online_gp_internal_viz_v5_dim_X.png")
            #         print("图像 online_gp_internal_viz_v5_dim_X.png 已保存 (captured active figure).")
            #     else:
            #         print("内部可视化函数被调用，但没有活动的Matplotlib图像可供保存。")
            # except Exception as e:
            #     print(f"尝试保存内部可视化图像时出错: {e}")
        else:
            print("维度0的GP模型尚未准备好进行内部可视化 (可能未训练或无数据)。")
    else:
        print("GP模型列表为空或第一个模型未初始化，无法进行内部可视化。")

    print("测试脚本执行完毕。")
    if plt.get_fignums():
        print("请关闭所有Matplotlib图像窗口以结束脚本。")
        plt.show() # Keep this commented out if running in a non-interactive environment
                     # or if visualize_single_dimension already calls plt.show()

if __name__ == '__main__':
    main_test_online_gp()
