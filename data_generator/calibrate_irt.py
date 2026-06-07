"""
data_generator/calibrate_irt.py
---------------------------------
Step 2 — Estimate IRT item difficulties from historical interactions.

Reads   big_data/interactions.parquet  +  big_data/tasks.parquet
Writes  big_data/irt_params.parquet

Run:  python data_generator/calibrate_irt.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import src.data_config as config
from src.spark_utils import get_spark
from src.irt_model import calibrate_item_difficulties


def main():
    spark = get_spark("IRT-Calibration", driver_memory="6g", shuffle_partitions=100)
    spark.sparkContext.setLogLevel("WARN")

    print("Calibrating IRT item difficulties …")
    irt_pd = calibrate_item_difficulties(spark, config.DATA_PATH, config.TASKS_PATH)

    print(f"\nCalibrated {len(irt_pd)} tasks")
    print(irt_pd[["task_id", "difficulty_raw", "difficulty_irt",
                   "p_correct_empirical", "n_interactions"]].describe())

    os.makedirs("big_data", exist_ok=True)
    irt_pd.to_parquet(config.IRT_PARAMS_PATH, index=False)
    print(f"\nSaved → {config.IRT_PARAMS_PATH}")

    corr = np.corrcoef(
        irt_pd["difficulty_norm"].fillna(0),
        irt_pd["difficulty_irt"].fillna(0),
    )[0, 1]
    print(f"Pearson r (sim difficulty vs IRT difficulty): {corr:.4f}")
    spark.stop()


if __name__ == "__main__":
    main()
