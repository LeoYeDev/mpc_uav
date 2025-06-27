import torch
import gpytorch
import time
import numpy as np
import copy
from collections import deque
import matplotlib.pyplot as plt

# 导入多进程和队列相关模块
from multiprocessing import Process, Queue, Event
import queue  # 用于处理空队列异常
import traceback # 用于打印详细的错误信息

# =================================================================================
# 1. 模型与数据缓冲区定义
# =================================================================================
class ExactGPModel(gpytorch.models.ExactGP):
    """
    标准的精确高斯过程模型定义。
    我们使用一个常量均值函数和一个带有缩放核的Matern核函数。
    Matern核通常在物理系统中比RBF核表现得更稳定。

    A standard Exact GP model, using a Matern kernel which is often more stable
    for physical systems than an RBF kernel.
    """
    def __init__(self, train_x, train_y, likelihood, ard_num_dims=1):
        # 初始化父类
        super().__init__(train_x, train_y, likelihood)
        # 定义均值函数为常量
        self.mean_module = gpytorch.means.ConstantMean()
        # 定义协方差函数（核函数）
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        )
    
    # 定义模型的前向传播
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # 返回一个多元高斯分布
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultiLevelBuffer:
    """
    多级数据缓冲区。
    - 负责高效地存储和采样数据点。
    - 拥有多个层级，每个层级有不同的容量和稀疏度。
    - 新数据优先进入最高密度层，当层满时，最旧的数据会被“挤出”并尝试进入下一稀疏层。
    """
    def __init__(self, level_max_capacities, level_sparsity_factors):
        # 检查输入合法性
        if len(level_max_capacities) != len(level_sparsity_factors):
            raise ValueError("Capacities and sparsity factor lists must have the same length.")
        self.num_levels = len(level_max_capacities)
        self.capacities = level_max_capacities
        # 创建每一层的双端队列作为缓冲区
        self.levels = [deque(maxlen=cap) for cap in self.capacities]
        self.sparsity_factors = level_sparsity_factors
        # 记录总共添加的数据点数量
        self.total_adds = 0

    def insert(self, v_scalar, y_scalar):
        """
        向缓冲区中插入一个新的数据点 (v, y)。
        """
        self.total_adds += 1
        item_to_process = (v_scalar, y_scalar)

        for i in range(self.num_levels):
            if item_to_process is None:
                break

            # 根据稀疏因子判断当前层是否接收此数据点
            if self.sparsity_factors[i] > 0 and self.total_adds % self.sparsity_factors[i] == 0:
                current_level_deque = self.levels[i]
                
                # 如果缓冲区已满，挤出最旧的数据点，传递给下一层处理
                next_item_to_process = current_level_deque.popleft() if len(current_level_deque) == current_level_deque.maxlen else None
                current_level_deque.append(item_to_process)
                item_to_process = next_item_to_process
            else:
                # 如果不满足稀疏条件，此数据点原封不动地传递给下一层
                pass

    def get_training_set(self):
        """
        获取用于训练的完整数据集。
        它会合并所有层级的数据并去除重复项。
        """
        all_data = []
        for level_deque in self.levels:
            all_data.extend(list(level_deque))
        # 使用字典去重，同时保持数据点的插入顺序
        return list(dict.fromkeys(all_data))
    
    def reset(self):
        """清空所有内部缓冲区。"""
        # *** 核心修复 2: 使用正确的属性名 (self.levels) 和 (self.capacities) ***
        self.levels = [deque(maxlen=cap) for cap in self.capacities]
        # *** 核心修复 3: 同时重置数据点计数器，确保完全重置 ***
        self.total_adds = 0
        print("  - MultiLevelBuffer has been reset.")
# =================================================================================
# 2. 训练历史记录器
# =================================================================================
class TrainingHistory:
    """一个简单的数据类，用于存储单次训练任务中超参数的演化历史。"""
    def __init__(self):
        self.loss = []
        self.noise = []
        self.lengthscale = []
        self.outputscale = []
        self.learning_rate = []

    def record(self, loss, model, optimizer):
        self.loss.append(loss)
        self.noise.append(model.likelihood.noise.item())
        if isinstance(model.covar_module.base_kernel, gpytorch.kernels.MaternKernel):
            self.lengthscale.append(model.covar_module.base_kernel.lengthscale.item())
        self.outputscale.append(model.covar_module.outputscale.item())
        self.learning_rate.append(optimizer.param_groups[0]['lr'])

    def has_data(self):
        return len(self.loss) > 0

# =================================================================================
# 3. 后台工作进程函数
# =================================================================================
def gp_training_worker(
    task_queue: Queue, 
    result_queue: Queue, 
    stop_event, 
    worker_config: dict, 
    dim_idx: int
):
    """
    后台工作进程函数。
    - 这是一个独立的进程，专门用于执行耗时的GP超参数优化。
    - 它循环地从自己的任务队列中获取任务，训练模型，然后将结果放入公共的结果队列。
    """
    device_str = worker_config.get('device_str', 'cpu')
    device = torch.device(device_str)
    
    # 初始化一个仅属于此进程的模型、似然和边际对数似然(mll)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    dummy_x = torch.zeros(2, 1, device=device)
    dummy_y = torch.zeros(2, device=device)
    model = ExactGPModel(dummy_x, dummy_y, likelihood, ard_num_dims=1).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    while not stop_event.is_set():
        try:
            # 使用超时来阻塞式获取任务，这样可以周期性地检查停止事件
            task = task_queue.get(timeout=0.2) 
            if task is None: # 收到“毒丸”(None)，表示主进程要求退出
                break
            
            train_x_tensor, train_y_tensor, initial_state_dict = task
            
            # 加载数据和模型状态到指定设备
            train_x_tensor = train_x_tensor.to(device)
            train_y_tensor = train_y_tensor.to(device)
            model.load_state_dict(initial_state_dict['model'])
            likelihood.load_state_dict(initial_state_dict['likelihood'])

            # [关键步骤] 将训练数据与模型关联起来
            model.set_train_data(inputs=train_x_tensor, targets=train_y_tensor, strict=False)
            
            # --- 开始执行完整的、阻塞式的训练 ---
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=worker_config.get('lr', 0.01))
            history = TrainingHistory()
            
            start_time = time.time()
            for i in range(worker_config.get('n_iter', 100)):
                optimizer.zero_grad()
                output = model(train_x_tensor)
                loss = -mll(output, train_y_tensor)
                loss.backward()
                optimizer.step()
                history.record(loss.item(), model, optimizer)
            duration = time.time() - start_time
            
            # [最佳实践] 将模型状态移回CPU，再打包结果
            model.cpu()
            likelihood.cpu()
            new_state_dict = {
                'model': model.state_dict(),
                'likelihood': likelihood.state_dict(),
            }
            model.to(device) # 移回工作设备，以备下次任务

            # 将带有维度标识符的结果放入共享的结果队列
            result_queue.put((dim_idx, new_state_dict, history, duration))

        except queue.Empty:
            # 队列为空是正常情况，继续循环
            continue
        except Exception as e:
            print(f"[Worker-{dim_idx}] 错误: 训练过程中发生致命错误: {e}")
            traceback.print_exc()

# =================================================================================
# 4. 主进程中的GP实例 (状态持有者)
# =================================================================================
class IncrementalGP:
    """
    增量高斯过程实例。
    - 在主进程中运行，每个维度一个。
    - 它本身不执行耗时的训练，只负责持有最新的模型状态、管理数据缓冲区和归一化统计量。
    """
    def __init__(self, dim_idx, config):
        self.dim_idx = dim_idx
        self.config = config
        self.device = torch.device(config.get('main_process_device', 'cpu'))
        self.epsilon = 1e-7
        
        # 数据缓冲区
        self.buffer = MultiLevelBuffer(config['buffer_level_capacities'], config['buffer_level_sparsity'])
        # --- 新增：一个用于存储所有历史数据的完整缓冲区 (无稀疏) ---
        # capacities = np.array(config['buffer_level_capacities'])
        # sparsity = np.array(config['buffer_level_sparsity'])
        # max_history_size = int(np.sum(capacities * sparsity))
        # self.full_history_buffer = deque(maxlen=max_history_size)

        # EMA (指数移动平均) 归一化统计量
        self.ema_alpha = config.get('ema_alpha', 0.05)
        self.v_mean_ema, self.v_var_ema = 0.0, 1.0
        self.r_mean_ema, self.r_var_ema = 0.0, 1.0
        
        # 主进程中用于快速预测的“实时”模型
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        dummy_x = torch.zeros(2, 1, device=self.device)
        dummy_y = torch.zeros(2, device=self.device)
        self.model = ExactGPModel(dummy_x, dummy_y, self.likelihood, ard_num_dims=1).to(self.device)
        
        # 状态标志
        self.updates_since_last_train = 0  # 距离上次训练有多少次数据更新
        self.is_trained_once = False       # 是否至少被成功训练过一次
        self.is_training_in_progress = False # 后台是否正在为此维度进行训练
        self.last_training_history = TrainingHistory() # 保存最近一次的训练历史

    def add_data_point(self, x, y):
        # --- 修改：同时向两个缓冲区添加数据 ---
        # self.full_history_buffer.append((x, y)) # 1. 记录到完整历史中
        self.buffer.insert(x, y)
        self.updates_since_last_train += 1
    
    def get_and_normalize_data(self):
        """获取并归一化所有数据，用于派发给后台训练和更新主模型。"""
        training_data_raw = self.buffer.get_training_set()
        if not training_data_raw or len(training_data_raw) < 2: return None, None
        
        raw_v = np.array([p[0] for p in training_data_raw], dtype=np.float32)
        raw_r = np.array([p[1] for p in training_data_raw], dtype=np.float32)
        
        # 更新EMA统计量
        num_points = len(raw_v)
        if num_points < self.config.get('min_points_for_ema', 30):
            v_mean, v_var = np.mean(raw_v), np.var(raw_v)
            r_mean, r_var = np.mean(raw_r), np.var(raw_r)
        else:
            self.v_mean_ema = (1 - self.ema_alpha) * self.v_mean_ema + self.ema_alpha * np.mean(raw_v)
            self.v_var_ema = (1 - self.ema_alpha) * self.v_var_ema + self.ema_alpha * np.var(raw_v)
            self.r_mean_ema = (1 - self.ema_alpha) * self.r_mean_ema + self.ema_alpha * np.mean(raw_r)
            self.r_var_ema = (1 - self.ema_alpha) * self.r_var_ema + self.ema_alpha * np.var(raw_r)
            v_mean, v_var = self.v_mean_ema, self.v_var_ema
            r_mean, r_var = self.r_mean_ema, self.r_var_ema
        
        # 使用统计量进行归一化
        train_x_norm = torch.tensor((raw_v - v_mean) / np.sqrt(v_var + self.epsilon), device=self.device).view(-1, 1)
        train_y_norm = torch.tensor((raw_r - r_mean) / np.sqrt(r_var + self.epsilon), device=self.device)
        return train_x_norm, train_y_norm

    def get_current_state_for_worker(self):
        """获取当前模型的状态字典，准备发送给worker。"""
        self.model.cpu()
        self.likelihood.cpu()
        state = {'model': self.model.state_dict(), 'likelihood': self.likelihood.state_dict()}
        self.model.to(self.device)
        return state

    def load_new_state_from_worker(self, new_state_dict, history):
        """从worker加载训练好的新状态字典并更新实时模型。"""
        self.model.load_state_dict(new_state_dict['model'])
        self.likelihood.load_state_dict(new_state_dict['likelihood'])
        self.model.to(self.device)
        
        # [关键步骤] 加载新超参数后，必须立即用最新的全量数据更新模型，否则预测会使用旧数据
        train_x_norm, train_y_norm = self.get_and_normalize_data()
        if train_x_norm is not None:
             self.model.set_train_data(train_x_norm, train_y_norm, strict=False)

        # 更新状态标志
        self.last_training_history = history
        self.is_training_in_progress = False
        self.updates_since_last_train = 0
        if not self.is_trained_once: self.is_trained_once = True
    
    def reset(self):
        """将此GP实例完全重置到其初始状态。"""
        # 1. 重置数据缓冲区
        self.buffer.reset()

        # 2. 重置所有状态标志
        self.is_trained_once = False
        self.is_training_in_progress = False
        self.updates_since_last_train = 0
        # 修复: 重置为一个新的空历史对象，而不是None
        self.last_training_history = TrainingHistory()

        # 3. 重置归一化统计数据 (EMA)
        self.v_mean_ema = 0.0
        self.v_var_ema = 1.0
        self.r_mean_ema = 0.0
        self.r_var_ema = 1.0
        # self.ema_counter = 0 # 移除未使用的变量

        # 4. *** 核心修复: 以正确的顺序重新初始化模型和似然 ***
        #    必须先创建新的likelihood，再用它来创建新的model。
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(None, None, self.likelihood).to(self.device)
        
        print(f"  - IncrementalGP for Dim has been fully reset.")

# =================================================================================
# 5. 管理器：负责编排所有组件
# =================================================================================
class IncrementalGPManager:
    """
    在线高斯过程管理器。
    - 负责创建和管理所有后台工作进程和通信队列。
    - 提供统一的接口 (`update`, `predict`, `shutdown`) 给外部调用。
    """
    def __init__(self, config):
        self.config = config
        self.num_dimensions = config.get('num_dimensions', 3)
        self.gps = [IncrementalGP(i, config) for i in range(self.num_dimensions)]
        
        # 为每个worker创建一个专属的任务队列
        self.task_queues = [Queue() for _ in range(self.num_dimensions)]
        # 所有worker共享一个公共的结果队列
        self.result_queue = Queue()
        # 用于通知所有worker停止的事件
        self.stop_event = Event()
        self.workers = []
        
        # 新增: 用于存储后台训练耗时的列表
        self.training_durations = []
        
        print(f"[管理器] 初始化... 正在为 {self.num_dimensions} 个维度启动后台工作进程。")
        for i in range(self.num_dimensions):
            worker_config = {
                'n_iter': config.get('worker_train_iters', 150),
                'lr': config.get('worker_lr', 0.01),
                'device_str': config.get('worker_device_str', 'cpu'),
            }
            worker = Process(target=gp_training_worker, args=(self.task_queues[i], self.result_queue, self.stop_event, worker_config, i))
            worker.daemon = True # 设置为守护进程，主进程退出时它会自动终止
            worker.start()
            self.workers.append(worker)
        print("[管理器] 所有后台工作进程已成功启动。")

    def update(self, new_velocities, new_residuals):
        """非阻塞更新：添加数据点，并根据条件触发后台训练任务。"""
        for i in range(self.num_dimensions):
            gp = self.gps[i]
            gp.add_data_point(new_velocities[i], new_residuals[i])
            
            if gp.is_training_in_progress: 
                continue
            
            # 检查是否满足触发训练的条件
            num_training_points = len(gp.buffer.get_training_set())
            should_trigger = False
            # 条件1: 首次训练
            if not gp.is_trained_once and num_training_points >= self.config.get('min_points_for_initial_train', 50):
                should_trigger = True
            # 条件2: 定期再训练
            elif gp.is_trained_once and gp.updates_since_last_train >= self.config.get('refit_hyperparams_interval', 100):
                should_trigger = True
            
            if should_trigger:
                train_x, train_y = gp.get_and_normalize_data()
                if train_x is not None:
                    print(f"🧠 [管理器] Dim-{i}: 满足训练条件 ({num_training_points}个点), 正在派发任务至后台...")
                    current_state = gp.get_current_state_for_worker()
                    task = (train_x, train_y, current_state)
                    self.task_queues[i].put(task)
                    gp.is_training_in_progress = True

    def poll_for_results(self):
        """非阻塞地检查并应用已完成的训练结果。"""
        try:
            dim_idx, new_state_dict, history, duration = self.result_queue.get_nowait()
            print(f"🎉 [管理器] 收到 Worker-{dim_idx} 的训练结果！耗时: {duration:.2f}s。正在更新实时模型...")
            # 新增: 记录训练时长
            self.training_durations.append(duration)
            self.gps[dim_idx].load_new_state_from_worker(new_state_dict, history)
        except queue.Empty:
            pass # 队列为空是正常情况

    def shutdown(self):
        """优雅地关闭所有后台工作进程。"""
        print("\n gracefully shutting down all background worker processes...")
        self.stop_event.set()
        for q in self.task_queues:
            q.put(None)
            
        time.sleep(0.5)
        for i, worker in enumerate(self.workers):
            worker.join(timeout=2.0)
            if worker.is_alive():
                print(f"[管理器] Worker-{i} 未能正常关闭，将强制终止。")
                worker.terminate()
        print("[管理器] 所有后台工作进程已成功关闭。")
    
    def _clear_queue(self, q):
        """安全地清空一个多进程队列。"""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def reset(self):
       """
       重置管理器及其所有内部GP实例的状态，为一次新的独立实验运行做准备。
       这会清空所有数据缓冲区、重置模型、并清空所有通信队列。
       """
       
       print("\n🔄 Resetting IncrementalGPManager state for new experiment run...")
       
       # 1. 委托每个GP实例进行重置 (清空缓冲区, 重置模型和状态)
       for gp in self.gps:
           gp.reset()
           gp.is_trained_once = False
           
       # 2. *** 核心修复: 清空所有任务队列和结果队列 ***
       print("  - Clearing communication queues...")
       for q in self.task_queues:
           self._clear_queue(q)
       self._clear_queue(self.result_queue)
       
       # 3. *** 核心修复: 重置性能统计列表 ***
       self.training_durations = []
       
       print("✅ Manager reset complete.") 

    def predict(self, query_velocities):
        """使用主进程中的实时模型进行预测。"""
        means = np.zeros((query_velocities.shape[0], self.num_dimensions), dtype=float)
        variances = np.ones_like(means, dtype=float)

        for i, gp in enumerate(self.gps):
            if not gp.is_trained_once: continue
            
            v_query = np.atleast_1d(query_velocities[:, i])
            if v_query.size == 0: continue
            
            v_std_ema = np.sqrt(gp.v_var_ema + gp.epsilon)
            v_query_norm = (v_query - gp.v_mean_ema) / v_std_ema
            # 最终修复：通过访问模型的一个参数来安全地获取其dtype
            model_dtype = next(gp.model.parameters()).dtype
            v_query_torch = torch.tensor(v_query_norm, device=gp.device, dtype=model_dtype).view(-1, 1)
            
            gp.model.eval()
            gp.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = gp.likelihood(gp.model(v_query_torch))
                mean_norm = preds.mean.cpu().numpy()
                var_norm = preds.variance.cpu().numpy()
            
            r_std_ema = np.sqrt(gp.r_var_ema + gp.epsilon)
            means[:, i] = mean_norm * r_std_ema + gp.r_mean_ema
            variances[:, i] = var_norm * (gp.r_var_ema + gp.epsilon)
            
        return means, np.maximum(variances, 1e-9)

    # =================================================================================
    # 可视化部分 (美化和改进)
    # =================================================================================
    def visualize_all_dims(self, sim_velocities, sim_residuals):
        """可视化所有维度的最终拟合结果。"""
        if not any(gp.is_trained_once for gp in self.gps):
            print("[可视化] 无法绘制拟合图，没有任何GP被训练过。")
            return
        
        # 修复: 使用兼容性更好的样式名
        try:
            plt.style.use('seaborn-whitegrid')
        except IOError:
            print("⚠️ [可视化] 'seaborn-whitegrid' style not found. Using default style.")
            plt.style.use('default')

        fig, axes = plt.subplots(1, self.num_dimensions, figsize=(24, 7), dpi=120)
        # 英文总标题
        fig.suptitle("Asynchronous GP Regression - Final Fit", fontsize=20, weight='bold')

        for i, (ax, gp) in enumerate(zip(axes, self.gps)):
            if not gp.is_trained_once: 
                ax.set_title(f'Dimension {i} - Not Trained Yet', style='italic')
                continue
            
            v_all_data = sim_velocities[:, i]
            r_all_data = sim_residuals[:, i]
            # 绘制所有历史数据点作为背景
            ax.plot(v_all_data, r_all_data, 'o', color='lightgrey', markersize=3, alpha=0.5, label='Full History Data')

            # 获取并突出显示当前缓冲区中的数据点
            current_buffer_data = gp.buffer.get_training_set()
            if current_buffer_data:
                v_buffer = [p[0] for p in current_buffer_data]
                r_buffer = [p[1] for p in current_buffer_data]
                ax.plot(v_buffer, r_buffer, 'o', color='#3498db', markersize=4, label='Active Buffer Data')

            # 准备用于绘制GP预测曲线的查询点
            v_range = np.linspace(v_all_data.min(), v_all_data.max(), 200)
            query_points = np.zeros((200, self.num_dimensions))
            for j in range(self.num_dimensions):
                query_points[:, j] = v_range if i == j else self.gps[j].v_mean_ema

            pred_mean, pred_var = self.predict(query_points)
            pred_std = np.sqrt(pred_var[:, i])
            
            # 绘制GP均值预测和置信区间
            ax.plot(v_range, pred_mean[:, i], color='#e74c3c', linestyle='-', linewidth=2.5, label='GP Mean Prediction')
            ax.fill_between(
                v_range,
                pred_mean[:, i] - 1.96 * pred_std,
                pred_mean[:, i] + 1.96 * pred_std,
                color='#e74c3c', alpha=0.2, label='95% Confidence Interval'
            )
            # 设置英文标题和坐标轴
            ax.set_title(f'Dimension {i} Fit', fontsize=16)
            ax.set_xlabel(f'Input: Velocity Dim {i} (m/s)', fontsize=12)
            ax.set_ylabel(f'Output: Residual Dim {i} (m/s^2)', fontsize=12)
            ax.legend()
            ax.set_xlim(v_all_data.min(), v_all_data.max())

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)

    def visualize_training_history(self):
        """可视化最近一次完成的训练任务中的超参数演化。"""
        if not any(gp.last_training_history.has_data() for gp in self.gps):
            print("[可视化] 无法绘制训练历史，还没有任何训练完成。")
            return
        
        # 修复: 使用兼容性更好的样式名
        try:
            plt.style.use('seaborn-whitegrid')
        except IOError:
            print("[可视化] 'seaborn-whitegrid' style not found. Using default style.")
            plt.style.use('default')
            
        fig, axes = plt.subplots(self.num_dimensions, 4, figsize=(20, 4 * self.num_dimensions), sharex=True, squeeze=False, dpi=120)
        fig.suptitle("Hyperparameter Evolution - Last Completed Training Task", fontsize=20, weight='bold')

        for i, gp in enumerate(self.gps):
            history = gp.last_training_history
            if not history.has_data():
                for j in range(4):
                   axes[i, j].text(0.5, 0.5, 'Not Trained Yet', ha='center', va='center', style='italic')
                   axes[i, j].set_title(f"Dim {i} - No History")
                continue
            
            iters = range(len(history.loss))
            param_names = ['Loss', 'Likelihood Noise', 'Kernel Lengthscale', 'Kernel Outputscale']
            data_to_plot = [history.loss, history.noise, history.lengthscale, history.outputscale]

            for j, (name, data) in enumerate(zip(param_names, data_to_plot)):
                ax = axes[i, j]
                ax.plot(iters, data, marker='.', linestyle='-', markersize=4)
                ax.set_title(f"Dim {i}: {name}", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)
                if i == self.num_dimensions - 1:
                    ax.set_xlabel("Optimization Iteration", fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)

# =================================================================================
#  6. 示例运行
# =================================================================================
if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except (RuntimeError, ValueError):
        pass # 在非Linux系统上或特定环境中可能会失败，属于正常情况

    # 定义GP管理器和仿真的配置
    final_config = {
        'num_dimensions': 3,
        'main_process_device': 'cuda',
        'worker_device_str': 'cuda',
        'buffer_level_capacities': [10, 15, 20], # 三层缓冲区容量
        'buffer_level_sparsity': [1, 2, 5],      # 稀疏因子：每1/2/5个点存入
        'min_points_for_initial_train': 30,      # 触发首次训练的最小数据点
        'min_points_for_ema': 30,                # 启用EMA所需的最小数据点
        'refit_hyperparams_interval': 20,       # 触发再训练的更新次数间隔
        'worker_train_iters': 70,               # 后台训练迭代次数
        'worker_lr': 0.03,                       # 训练学习率
        'ema_alpha': 0.05,                       # EMA平滑系数
    }
    
    manager = IncrementalGPManager(config=final_config)
    
    # --- 生成模拟数据 ---
    num_total_samples = 1200
    time_signal = np.linspace(0, 120, num_total_samples)
    sim_velocities = np.zeros((num_total_samples, manager.num_dimensions))
    sim_residuals = np.zeros((num_total_samples, manager.num_dimensions))
    
    sim_velocities[:, 0] = 1.0 * np.sin(0.2 * time_signal) + 0.5 * np.cos(0.5 * time_signal + 0.5)
    sim_residuals[:, 0] = (0.3 * np.sin(2.5 * sim_velocities[:, 0]) + 0.05 * sim_velocities[:, 0] + 0.02 * np.random.randn(num_total_samples))
    sim_velocities[:, 1] = 0.8 * np.sin(0.25 * time_signal + 1.0) + 0.3 * np.sin(0.6 * time_signal)
    sim_residuals[:, 1] = (0.4 * np.exp(-15.0 * (sim_velocities[:, 1] - 0.5)**2) + 0.2 * np.cos(3.0 * sim_velocities[:, 1]) + 0.03 * np.random.randn(num_total_samples))
    sim_velocities[:, 2] = 0.6 * np.sin(0.3 * time_signal + 2.0) - 0.2 * (time_signal / 60)
    sim_residuals[:, 2] = (0.2 * sim_velocities[:, 2]**2 - 0.1 * sim_velocities[:, 2] + 0.05 + 0.025 * np.random.randn(num_total_samples))

    # --- 运行主仿真循环 ---
    print("\n" + "="*50)
    print("开始运行异步训练仿真...")
    print("="*50)
    main_loop_times = []
    
    try:
        for i in range(num_total_samples):
            start_t = time.perf_counter()
            
            # 1. (非阻塞) 派发新数据和可能的训练任务
            manager.update(sim_velocities[i, :], sim_residuals[i, :])
            
            # 2. (非阻塞) 检查是否有训练好的结果需要应用
            manager.poll_for_results()
            
            main_loop_times.append((time.perf_counter() - start_t) * 1000)

            # 3. 模拟控制周期的其余工作负载 (例如，10ms)
            time.sleep(0.01)
            
            if (i + 1) % 200 == 0:
                print(f"  [进度] 仿真步骤 {i+1}/{num_total_samples} 已处理...")

    except KeyboardInterrupt:
        print("\n用户中断仿真。")
    finally:
        # 确保无论如何都优雅地关闭后台进程
        manager.shutdown()

    # --- 性能统计与可视化 ---
    main_loop_times = np.array(main_loop_times)
    print("\n" + "="*50)
    print("主循环性能统计 (不含后台训练耗时)")
    print("="*50)
    print(f"  - 平均循环耗时: {np.mean(main_loop_times):.4f} ms")
    print(f"  - 中位数耗时:   {np.median(main_loop_times):.4f} ms")
    print(f"  - 最大循环耗时:   {np.max(main_loop_times):.2f} ms")
    print(f"  - 99百分位耗时: {np.percentile(main_loop_times, 99):.2f} ms")
    
    # 新增: 后台训练耗时统计
    if manager.training_durations:
        durations = np.array(manager.training_durations)
        print("\n" + "="*50)
        print("后台训练任务耗时统计")
        print("="*50)
        print(f"  - 完成的训练任务总数: {len(durations)}")
        print(f"  - 平均训练耗时: {np.mean(durations):.2f} s")
        print(f"  - 最短训练耗时: {np.min(durations):.2f} s")
        print(f"  - 最长训练耗时: {np.max(durations):.2f} s")


    print("\n" + "="*50)
    print("正在生成可视化图表...")
    print("="*50)
    manager.visualize_all_dims(sim_velocities, sim_residuals)
    manager.visualize_training_history()
    
    # 绘制主循环时间分布图
    # 修复: 使用兼容性更好的样式名
    try:
        plt.style.use('seaborn-whitegrid')
    except IOError:
        plt.style.use('default')
        
    plt.figure(figsize=(12, 7), dpi=100)
    plt.hist(main_loop_times, bins=50, alpha=0.75, color='#3498db', label='Update Times')
    plt.axvline(np.mean(main_loop_times), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {np.mean(main_loop_times):.2f} ms')
    plt.axvline(np.median(main_loop_times), color='#f1c40f', linestyle=':', linewidth=2.5, label=f'Median: {np.median(main_loop_times):.2f} ms')
    plt.xlabel("Main Loop Update Time (ms)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Main Loop Update Times", fontsize=16, weight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

