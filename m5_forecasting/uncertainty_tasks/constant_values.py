import numpy as np

PERCENTILE_LOWERS = np.array([0.005, 0.025, 0.165, 0.25])
PERCENTILE_UPPERS = np.array([0.75, 0.835, 0.975, 0.995])
PERCENTILES = np.concatenate([PERCENTILE_LOWERS, np.array([0.5]), PERCENTILE_UPPERS])

LEVELS = [["_all_"], ["id"], ["item_id"], ["dept_id"], ["cat_id"], ["store_id"], ["state_id"],
          ["state_id", "item_id"], ["state_id", "dept_id"], ["store_id", "dept_id"], ["state_id", "cat_id"],
          ["store_id", "cat_id"]]

COLS = [f"F{i}" for i in range(1, 29)]
