
import unittest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.gp.buffer import InformationGainBuffer, FIFOBuffer, MultiLevelInformationGainBuffer

class TestBuffers(unittest.TestCase):
    
    def test_fifo_buffer(self):
        buf = FIFOBuffer(max_size=3)
        buf.insert(1.0, 1.0, 0.1)
        buf.insert(2.0, 1.0, 0.2)
        buf.insert(3.0, 1.0, 0.3)
        
        self.assertEqual(len(buf.get_data()[0]), 3)
        
        # Should remove oldest (1.0)
        buf.insert(4.0, 1.0, 0.4)
        data_v, _ = buf.get_data()
        self.assertTrue(4.0 in data_v)
        self.assertFalse(1.0 in data_v)
        self.assertTrue(2.0 in data_v)

    def test_ivs_buffer_novelty(self):
        """Test if IVS prefers novel points."""
        buf = InformationGainBuffer(max_size=3, novelty_weight=1.0, recency_weight=0.0, min_distance=0.0)
        
        buf.insert(1.0, 0.0, 0.0)
        buf.insert(1.1, 0.0, 0.0)
        buf.insert(1.2, 0.0, 0.0)
        
        buf.insert(5.0, 0.0, 0.0)
        data_v, _ = buf.get_data()
        self.assertTrue(5.0 in data_v) 
        self.assertEqual(len(data_v), 3)

    def test_ivs_buffer_recency(self):
        """Test if IVS prefers recent points."""
        buf = InformationGainBuffer(max_size=3, novelty_weight=0.0, recency_weight=1.0, decay_rate=1.0, min_distance=0.0)
        
        buf.insert(1.0, 0.0, 0.0)
        buf.insert(2.0, 0.0, 0.1)
        buf.insert(3.0, 0.0, 0.2)
        
        buf.insert(4.0, 0.0, 10.0)
        data_v, _ = buf.get_data()
        
        self.assertTrue(4.0 in data_v) 
        self.assertFalse(1.0 in data_v) 

    def test_ivs_normalization(self):
        """Scores for symmetric points should be equal and finite."""
        buf = InformationGainBuffer(max_size=3)
        buf.insert(0.0, 0.0, 0.0)
        buf.insert(10.0, 0.0, 0.0)
        
        scores = buf._compute_scores()
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(scores[0], scores[1], places=7)

    def test_ivs_close_point_refreshes_residual(self):
        """Close samples should refresh stored residual, not keep stale y."""
        buf = InformationGainBuffer(max_size=4, min_distance=0.05)
        buf.insert(1.0, 0.1, 0.0)
        buf.insert(2.0, 0.2, 0.1)
        initial_len = len(buf.get_training_set())

        # Close in velocity space: should overwrite nearest sample with latest y,t.
        buf.insert(1.01, 9.9, 1.0)
        updated = buf.get_training_set()

        self.assertEqual(len(updated), initial_len)
        self.assertTrue(any(abs(v - 1.01) < 1e-9 and abs(y - 9.9) < 1e-9 for v, y in updated))

    def test_ivs_recency_uses_adaptive_time_scale(self):
        """Recency score should decay over sample steps, not be flat in wall-time seconds."""
        buf = InformationGainBuffer(max_size=6, novelty_weight=0.0, recency_weight=1.0, decay_rate=0.1)
        timestamps = np.array([0.00, 0.02, 0.04, 0.06], dtype=float)
        recency = buf._compute_recency_scores(timestamps)

        # newest ~= 1.0, oldest should show meaningful decay (exp(-0.3) ~= 0.7408)
        self.assertAlmostEqual(recency[-1], 1.0, places=6)
        self.assertLess(recency[0], 0.9)

    def test_multilevel_buffer_respects_global_capacity(self):
        buf = MultiLevelInformationGainBuffer(
            max_size=10,
            novelty_weight=0.4,
            recency_weight=0.6,
            level_capacities=[5, 3, 2],
            level_sparsity=[1, 2, 4],
        )
        for i in range(40):
            buf.insert(float(i) * 0.1, float(i), float(i) * 0.02)

        training = buf.get_training_set()
        self.assertLessEqual(len(training), 10)
        self.assertGreater(len(training), 0)

    def test_multilevel_buffer_keeps_novel_regions(self):
        buf = MultiLevelInformationGainBuffer(
            max_size=8,
            novelty_weight=0.7,
            recency_weight=0.3,
            min_distance=0.01,
            level_capacities=[4, 2, 2],
            level_sparsity=[1, 2, 3],
        )
        for i in range(15):
            buf.insert(1.0 + 0.005 * i, 0.0, 0.02 * i)
        buf.insert(3.5, 1.0, 1.0)  # Novel region spike

        v_vals = [v for v, _ in buf.get_training_set()]
        self.assertTrue(any(abs(v - 3.5) < 1e-9 for v in v_vals))

if __name__ == '__main__':
    unittest.main()
