import unittest
import numpy as np
import torch
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.gp.online import IncrementalGP
from config.gp_config import OnlineGPConfig, GPModelParams

class TestIncrementalGP(unittest.TestCase):
    def setUp(self):
        self.config = OnlineGPConfig(
            num_dimensions=1,
            worker_lr=0.01,
            buffer_level_capacities=[5, 10],
            buffer_level_sparsity=[1, 2]
        ).to_dict()
        
    def test_init(self):
        gp = IncrementalGP(dim_idx=0, config=self.config)
        self.assertIsNotNone(gp.model)
        self.assertIsNotNone(gp.likelihood)
        
    def test_get_model_params(self):
        gp = IncrementalGP(dim_idx=0, config=self.config)
        
        # Manually set some params to verify export
        with torch.no_grad():
            gp.model.covar_module.outputscale = torch.tensor(2.0)
            gp.likelihood.noise = torch.tensor(0.1)
            # lengthscale is usually initialized to something specific or random
        
        params = gp.get_model_params()
        
        self.assertIsInstance(params, GPModelParams)
        self.assertAlmostEqual(params.signal_variance, 2.0, places=4)
        self.assertAlmostEqual(params.noise_variance, 0.1, places=4)
        self.assertIsInstance(params.length_scale, list)
        self.assertIsInstance(params.mean, float)

if __name__ == '__main__':
    unittest.main()
