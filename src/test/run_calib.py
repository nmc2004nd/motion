import logging
import os
import sys
import numpy as np

# Add the project root to the python path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.config.loader import load_config
from src.core.calibration import calibrate_camera

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Check if pipeline_config.yaml is at the expected path
    config_path = os.path.join(project_root, "config", "pipeline_config.yaml")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        return
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return

    logging.info(f"Loaded config from {config_path}")
    logging.info("Starting camera calibration...")
    
    try:
        mtx, dist = calibrate_camera(config)
        if mtx is not None and dist is not None:
            logging.info("Calibration procedure finished successfully.")
            
            # Làm tròn để hiển thị cho gọn
            np.set_printoptions(precision=4, suppress=True)
            print("\n=== Kết quả hiệu chuẩn ===")
            print(f"Ma trận Camera (mtx):\n{np.round(mtx, 2)}")
            print(f"\nHệ số biến dạng (dist):\n{np.round(dist, 5)}")
            print("==========================\n")
        else:
            logging.error("Calibration procedure failed.")
    except Exception as e:
        logging.error(f"Error during calibration: {e}")

if __name__ == "__main__":
    main()
