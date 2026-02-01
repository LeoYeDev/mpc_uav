import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_data_logger():
    print("Testing DataLogger...")
    try:
        from src.utils.data_logger import DataLogger
        
        logger = DataLogger()
        
        # Test basic logging
        logger.log("key1", 1)
        logger.log("key1", 2)
        logger.log("key2", np.array([1, 2, 3]))
        
        data = logger.to_dict()
        assert len(data['key1']) == 2
        assert np.array_equal(data['key2'][0], np.array([1, 2, 3]))
        print("Basic logging passed.")
        
        # Test save/pickle
        logger.save_pickle("tests/test_log.pkl")
        assert os.path.exists("tests/test_log.pkl")
        print("Pickle save passed.")
        
        # Cleanup
        os.remove("tests/test_log.pkl")
        
        print("DataLogger verification successful.")
    except Exception as e:
        print(f"DataLogger verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_data_logger()
