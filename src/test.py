import torch
import numpy as np
import matplotlib.pyplot as plt

capacities = np.array([11,7,2])
sparsity_factors = np.array([1,2,10])
cc = capacities * sparsity_factors
print(f"capacities: {capacities.tolist()}, sparsity_factors: {sparsity_factors}, cc: {cc}")
#输出各自数据类型是什么
print(f"capacities type: {capacities.dtype}, sparsity_factors type: {sparsity_factors.dtype}, cc type: {cc.dtype}")
# # ======================
# # 1. 生成合成测试数据（修复数据形状）
# # ======================
# def generate_data(n=100):
#     x = torch.linspace(0, 4*np.pi, n)
#     y = torch.sin(x) + 0.2 * torch.randn(n)  # y 形状为 [100]
#     return x.unsqueeze(-1), y  # x 形状为 [100, 1], y 形状为 [100]

# train_x, train_y = generate_data(100)
# test_x = torch.linspace(0, 4*np.pi, 200).unsqueeze(-1)

# # ======================
# # 2. 定义GP模型（无需修改）
# # ======================
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.RBFKernel()
        
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# # 初始化模型（确保 train_y 是 [100]）
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(train_x, train_y, likelihood)

# # ======================
# # 3. 训练过程（修复损失函数输入）
# # ======================
# model.train()
# likelihood.train()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# for i in range(20):
#     optimizer.zero_grad()
#     output = model(train_x)
#     loss = -mll(output, train_y)  # train_y 形状为 [100]，与 output 形状匹配
#     loss.backward()
#     optimizer.step()
#     print(f'Iter {i+2}/20 - Loss: {loss.item():.3f}')

# # ======================
# # 4. 预测和可视化（无需修改）
# # ======================
# model.eval()
# likelihood.eval()

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(test_x))

# # 可视化结果
# with torch.no_grad():
#     f, ax = plt.subplots(1, 1, figsize=(10, 6))
    
#     mean = observed_pred.mean.numpy()
#     lower, upper = observed_pred.confidence_region()
    
#     ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Training Data')
#     ax.plot(test_x.numpy(), mean, 'b', lw=2, label='Predicted Mean')
#     ax.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.2, color='blue', label='95% Confidence')
#     ax.plot(test_x.numpy(), torch.sin(test_x).numpy(), 'r--', lw=1, label='True Function')
    
#     ax.legend()
#     plt.show()