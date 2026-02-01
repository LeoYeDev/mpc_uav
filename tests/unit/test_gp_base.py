import unittest
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.gp.base import CustomKernelFunctions, CustomGPRegression
from config.gp_config import GPModelParams

class TestCustomKernelFunctions(unittest.TestCase):
    def test_init_defaults(self):
        kernel = CustomKernelFunctions('squared_exponential')
        self.assertEqual(kernel.params['l'], [1.0])
        self.assertEqual(kernel.params['sigma_f'], 1.0)

    def test_init_with_gp_model_params(self):
        params = GPModelParams(length_scale=[0.5, 0.5], signal_variance=2.0, noise_variance=0.1)
        kernel = CustomKernelFunctions('squared_exponential', params=params)
        np.testing.assert_array_equal(kernel.params['l'], np.array([0.5, 0.5]))
        self.assertAlmostEqual(kernel.params['sigma_f'], np.sqrt(2.0))

    def test_call_shape(self):
        kernel = CustomKernelFunctions('squared_exponential')
        x1 = np.random.rand(5, 3)
        x2 = np.random.rand(4, 3)
        k_mat = kernel(x1, x2)
        self.assertEqual(k_mat.shape, (5, 4))
        
    def test_call_diag(self):
        kernel = CustomKernelFunctions('squared_exponential')
        x1 = np.random.rand(5, 3)
        k_mat = kernel(x1, None) # should behave like kernel(x1, x1) with diag filled
        self.assertEqual(k_mat.shape, (5, 5))
        self.assertTrue(np.allclose(np.diag(k_mat), 1.0)) # sq exp kernel diag is sigma_f^2 = 1.0 default

class TestCustomGPRegression(unittest.TestCase):
    def setUp(self):
        self.gp = CustomGPRegression(x_features=[0, 1], u_features=[], reg_dim=0)
    
    def test_init(self):
        self.assertIsInstance(self.gp.kernel, CustomKernelFunctions)
        self.assertEqual(self.gp.reg_dim, 0)
        
    def test_load_and_predict(self):
        # Create a mock dictionary structure mimicking what is saved
        x_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        y_train = np.array([0.0, 1.0, 0.0])
        
        # We need mock K_inv, K_inv_y
        # For unit test, we don't need them to be mathematically perfect inverses, 
        # just compatible shapes for matmul in predict
        # K is 3x3
        k_inv = np.eye(3)
        # alpha is 3x1
        k_inv_y = np.ones(3)
        
        mock_data = {
            'x_train': x_train,
            'y_train': y_train,
            'k_inv': k_inv,
            'k_inv_y': k_inv_y,
            'kernel_type': 'squared_exponential',
            'kernel_params': {'l': np.array([1.0, 1.0]), 'sigma_f': 1.0},
            'sigma_n': 0.1,
            'mean': np.array([0.0, 0.0]),
            'y_mean': 0.0
        }
        
        self.gp.load(mock_data)
        
        # Test predict
        x_test = np.array([[0.5, 0.5]]) # 1 sample, 2 dims
        # x_features are [0, 1] so it expects input of dimension 2 (if we passed only x)
        # But predict takes 'x_test' which is usually the Feature Vector z.
        # CustomGPRegression.predict expects (n_samples, d_features) or (d_features, n_samples)? 
        # Let's check base.py predict: x_test = np.atleast_2d(x_test)... k_s = kernel(x_test, x_train)
        # Kernel expects (m, d). x_train is (3, 2). So x_test should be (n, 2).
        
        # Test 1 sample
        mu = self.gp.predict(x_test)
        self.assertTrue(isinstance(mu, np.ndarray))
        self.assertEqual(mu.shape, (1,)) 
        
        # Test return std
        mu, std = self.gp.predict(x_test, return_std=True)
        self.assertEqual(std.shape, (1,))

if __name__ == '__main__':
    unittest.main()
