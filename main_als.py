"""
main_als.py
-----------
Step 1 — Train ALS on interaction data and export artefacts.

What it produces
----------------
  big_data/als_user_factors.npz    ← user latent vectors  (numpy)
  big_data/als_item_factors.npz    ← item latent vectors  (numpy)
  big_data/results/metrics.parquet ← evaluation results at all K cutoffs

Run:  python main_als.py
"""

from pyspark.sql import functions as F
import numpy as np
import os

import src.data_config as config
from src.spark_utils import get_spark
from src.data_prep import prepare_for_als
from src.als_model import train_als
from src.evaluation import evaluate_all_k
from src.baselines import popularity_recs, random_recs
from src.data_loader import load_data, split_data


def save_factors(model, user_path: str, item_path: str) -> None:
    """Export ALS factor matrices as compressed numpy arrays for fast inference."""
    user_pd = model.userFactors.toPandas()
    item_pd = model.itemFactors.toPandas()

    user_ids     = user_pd["id"].to_numpy(dtype=np.int32)
    user_factors = np.vstack(user_pd["features"].to_numpy()).astype(np.float32)
    item_ids     = item_pd["id"].to_numpy(dtype=np.int32)
    item_factors = np.vstack(item_pd["features"].to_numpy()).astype(np.float32)

    os.makedirs("big_data", exist_ok=True)
    np.savez_compressed(user_path, ids=user_ids, factors=user_factors)
    np.savez_compressed(item_path, ids=item_ids, factors=item_factors)
    print(f"Saved user factors → {user_path}  {user_factors.shape}")
    print(f"Saved item factors → {item_path}  {item_factors.shape}")


def _make_recs(model, train_df, max_k: int):
    """Generate top-max_k recommendations, filtering out already-seen items."""
    raw_recs = model.recommendForAllUsers(max_k)
    seen     = train_df.groupBy(config.USER_COL).agg(F.collect_set(config.ITEM_COL).alias("seen"))
    return (
        raw_recs
        .join(seen, on=config.USER_COL, how="left")
        .withColumn(
            "recommendations",
            F.expr(f"filter(recommendations, r -> NOT array_contains(seen, r.{config.ITEM_COL}))")
        )
        .select(config.USER_COL, "recommendations")
        .cache()
    )


def main():
    spark = get_spark("ALS-Recommender")

    interactions = load_data(spark)
    df           = prepare_for_als(interactions).cache()
    df.count()

    train_df, test_df = split_data(df)
    train_df = train_df.cache(); test_df = test_df.cache()
    train_df.count(); test_df.count()

    model = train_als(train_df)

    # Export factors for adaptive engine
    save_factors(model, config.USER_FACTORS_PATH, config.ITEM_FACTORS_PATH)

    # Recommendations at max K (re-used for all cutoffs)
    max_k   = max(config.EVAL_K_LIST)
    als_recs = _make_recs(model, train_df, max_k)
    als_recs.count()

    pop_recs = popularity_recs(train_df, max_k).cache()
    rnd_recs = random_recs(train_df, max_k, seed=config.SEED).cache()

    # Evaluate at every K in EVAL_K_LIST 
    print(f"\n{'─'*70}")
    print(f"  {'Model':<12}  {'K':>4}  {'HR@K':>7}  {'Prec@K':>8}  {'Recall@K':>10}  {'MAP@K':>8}  {'NDCG@K':>8}")
    print(f"{'─'*70}")

    result_rows = []
    for k in config.EVAL_K_LIST:
        for name, recs in [("ALS", als_recs), ("Popularity", pop_recs), ("Random", rnd_recs)]:
            m = evaluate_all_k(recs, test_df, k)
            print(f"  {name:<12}  {k:>4}  {m['HR@K']:>7.4f}  {m['Precision@K']:>8.5f}"
                  f"  {m['Recall@K']:>10.6f}  {m.get('MAP@K', float('nan')):>8.5f}"
                  f"  {m.get('NDCG@K', float('nan')):>8.5f}")
            result_rows.append((name, k,
                                 m["HR@K"], m["Precision@K"], m["Recall@K"],
                                 m.get("MAP@K"), m.get("NDCG@K")))

    print(f"{'─'*70}\n")

    # Save results 
    os.makedirs(config.OUT_DIR, exist_ok=True)
    out = spark.createDataFrame(
        result_rows,
        ["model", "k", "hr_at_k", "precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"],
    )
    out.write.mode("overwrite").parquet(f"{config.OUT_DIR}/metrics.parquet")


if __name__ == "__main__":
    main()
