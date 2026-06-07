"""
tune_als.py
-----------
Grid search over ALS hyperparameters, evaluated with HR@K on a held-out
validation set.  Trains the best configuration on train+val and reports
final test metrics.

Split strategy: 70 % train / 15 % validation / 15 % test
Primary metric: HR@10  (most interpretable for adaptive learning use-case)

Grid searched:
  rank     — dimensionality of latent factors          [50, 100, 150]
  alpha    — confidence scaling for implicit feedback  [5, 10, 20, 40]
  regParam — L2 regularisation                         [0.01, 0.05, 0.1]

Total combinations: 36  (~2-4 min each on 20k×20k data → ~2-3 hours total)
Use --quick flag to run a reduced 12-combination grid (rank×alpha only).

Run:
    python tune_als.py           # full grid
    python tune_als.py --quick   # rank × alpha only (12 combos, ~1 hour)
"""

import argparse
import itertools
import os
import time

from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

import src.data_config as config
from src.spark_utils import get_spark
from src.data_prep import prepare_for_als
from src.data_loader import load_data
from src.evaluation import hitrate_precision_recall_at_k, map_at_k, ndcg_at_k


# Grid definitions

FULL_GRID = {
    "rank":     [50, 100, 150],
    "alpha":    [5, 10, 20, 40],
    "regParam": [0.01, 0.05, 0.1],
}

# Reduced grid: fix regParam at current best (0.05), search rank × alpha
QUICK_GRID = {
    "rank":     [50, 100, 150, 200],
    "alpha":    [5, 10, 20, 40],
    "regParam": [0.05],
}


# Helpers

def _recs(model, train_df, k):
    raw  = model.recommendForAllUsers(k)
    seen = train_df.groupBy(config.USER_COL).agg(F.collect_set(config.ITEM_COL).alias("seen"))
    return (
        raw
        .join(seen, on=config.USER_COL, how="left")
        .withColumn(
            "recommendations",
            F.expr(f"filter(recommendations, r -> NOT array_contains(seen, r.{config.ITEM_COL}))")
        )
        .select(config.USER_COL, "recommendations")
    )


def _train_eval(spark, train_df, val_df, params: dict, k: int) -> dict:
    als = ALS(
        userCol           = config.USER_COL,
        itemCol           = config.ITEM_COL,
        ratingCol         = config.RATING_COL,
        implicitPrefs     = True,
        nonnegative       = True,
        maxIter           = config.ALS_PARAMS["maxIter"],
        coldStartStrategy = "drop",
        **params,
    )
    model = als.fit(train_df)
    recs  = _recs(model, train_df, k)
    m     = hitrate_precision_recall_at_k(recs, val_df, k)
    m["MAP@K"]  = map_at_k(recs, val_df, k)
    m["NDCG@K"] = ndcg_at_k(recs, val_df, k)
    return m


def _print_row(rank, alpha, reg, m, elapsed, marker=""):
    print(
        f"  rank={rank:>3}  alpha={alpha:>4}  reg={reg:.3f}  │"
        f"  HR@10={m['HR@K']:.4f}  MAP@10={m['MAP@K']:.4f}"
        f"  NDCG@10={m['NDCG@K']:.4f}  [{elapsed:.0f}s] {marker}"
    )


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced grid (rank × alpha only).")
    args = parser.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID
    combos = list(itertools.product(grid["rank"], grid["alpha"], grid["regParam"]))
    print(f"\n{'═'*70}")
    print(f"  ALS Hyperparameter Grid Search — {len(combos)} combinations")
    print(f"  Primary metric: HR@{config.TOP_K}")
    print(f"  Split: 70% train / 15% val / 15% test")
    print(f"{'═'*70}\n")

    spark = get_spark("ALS-Tuning", driver_memory="6g", executor_memory="6g")
    spark.sparkContext.setLogLevel("WARN")

    interactions = load_data(spark)
    df = prepare_for_als(interactions).cache()
    df.count()

    # 70 / 15 / 15 split
    train_df, val_df, test_df = df.randomSplit([0.70, 0.15, 0.15], seed=config.SEED)
    train_df = train_df.cache()
    val_df   = val_df.cache()
    test_df  = test_df.cache()
    train_df.count(); val_df.count(); test_df.count()

    # Grid search
    best_hr     = -1.0
    best_params = {}
    best_m      = {}

    print(f"  {'rank':>4}  {'alpha':>5}  {'reg':>5}  │  HR@10   MAP@10  NDCG@10")
    print(f"  {'─'*60}")

    for i, (rank, alpha, reg) in enumerate(combos, 1):
        params = {"rank": rank, "alpha": float(alpha), "regParam": reg}
        t0 = time.time()
        try:
            m = _train_eval(spark, train_df, val_df, params, config.TOP_K)
        except Exception as e:
            print(f"  [SKIP] rank={rank} alpha={alpha} reg={reg} — {e}")
            continue
        elapsed = time.time() - t0

        marker = ""
        if m["HR@K"] > best_hr:
            best_hr     = m["HR@K"]
            best_params = params
            best_m      = m
            marker = "★ NEW BEST"

        _print_row(rank, alpha, reg, m, elapsed, marker)

    # Final evaluation with best params on train+val - test
    print(f"\n{'═'*70}")
    print(f"  Best params: {best_params}")
    print(f"  Val metrics: {best_m}")
    print(f"\n  Retraining on train + val, evaluating on held-out test …")
    print(f"{'═'*70}")

    # Combine train + val for final model
    train_val = train_df.union(val_df).cache()
    train_val.count()

    als_final = ALS(
        userCol           = config.USER_COL,
        itemCol           = config.ITEM_COL,
        ratingCol         = config.RATING_COL,
        implicitPrefs     = True,
        nonnegative       = True,
        maxIter           = config.ALS_PARAMS["maxIter"],
        coldStartStrategy = "drop",
        **best_params,
    )
    final_model  = als_final.fit(train_val)
    final_recs   = _recs(final_model, train_val, config.TOP_K)

    final_m = hitrate_precision_recall_at_k(final_recs, test_df, config.TOP_K)
    final_m["MAP@K"]  = map_at_k(final_recs, test_df, config.TOP_K)
    final_m["NDCG@K"] = ndcg_at_k(final_recs, test_df, config.TOP_K)

    print("  Evaluating baselines on the same test set …")
    from src.baselines import popularity_recs, random_recs

    pop_recs = popularity_recs(train_val, config.TOP_K).cache()
    rnd_recs = random_recs(train_val, config.TOP_K, seed=config.SEED).cache()

    pop_m = hitrate_precision_recall_at_k(pop_recs, test_df, config.TOP_K)
    pop_m["MAP@K"]  = map_at_k(pop_recs, test_df, config.TOP_K)
    pop_m["NDCG@K"] = ndcg_at_k(pop_recs, test_df, config.TOP_K)

    rnd_m = hitrate_precision_recall_at_k(rnd_recs, test_df, config.TOP_K)
    rnd_m["MAP@K"]  = map_at_k(rnd_recs, test_df, config.TOP_K)
    rnd_m["NDCG@K"] = ndcg_at_k(rnd_recs, test_df, config.TOP_K)

    k = config.TOP_K
    print(f"\n{'═'*70}")
    print(f"  THESIS TABLE — all models on the same held-out test set (70/15/15)")
    print(f"{'─'*70}")
    print(f"  {'Model':<12} {'HR@'+str(k):>7} {'Prec@'+str(k):>8} {'Recall@'+str(k):>10} {'MAP@'+str(k):>8} {'NDCG@'+str(k):>8}")
    print(f"  {'─'*65}")
    for name, m in [("ALS", final_m), ("Popularity", pop_m), ("Random", rnd_m)]:
        print(f"  {name:<12} {m['HR@K']:>7.4f} {m['Precision@K']:>8.5f}"
              f" {m['Recall@K']:>10.6f} {m['MAP@K']:>8.4f} {m['NDCG@K']:>8.4f}")
    print(f"{'═'*70}")

    # ALS lift over baselines
    print(f"\n  ALS lift over Popularity:  HR×{final_m['HR@K']/pop_m['HR@K']:.2f}"
          f"  MAP×{final_m['MAP@K']/pop_m['MAP@K']:.2f}"
          f"  NDCG×{final_m['NDCG@K']/pop_m['NDCG@K']:.2f}")
    print(f"  ALS lift over Random:      HR×{final_m['HR@K']/rnd_m['HR@K']:.2f}"
          f"  MAP×{final_m['MAP@K']/rnd_m['MAP@K']:.2f}"
          f"  NDCG×{final_m['NDCG@K']/rnd_m['NDCG@K']:.2f}")

    # Save tuning results
    os.makedirs(config.OUT_DIR, exist_ok=True)
    result = spark.createDataFrame(
        [
            (str(best_params), "ALS",        final_m["HR@K"], final_m["Precision@K"], final_m["Recall@K"], final_m["MAP@K"], final_m["NDCG@K"]),
            (str(best_params), "Popularity",  pop_m["HR@K"],   pop_m["Precision@K"],   pop_m["Recall@K"],   pop_m["MAP@K"],   pop_m["NDCG@K"]),
            (str(best_params), "Random",      rnd_m["HR@K"],   rnd_m["Precision@K"],   rnd_m["Recall@K"],   rnd_m["MAP@K"],   rnd_m["NDCG@K"]),
        ],
        ["best_params", "model", "hr_at_k", "precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"],
    )
    result.write.mode("overwrite").parquet(f"{config.OUT_DIR}/tuning_results.parquet")
    print(f"\n  Results saved → {config.OUT_DIR}/tuning_results.parquet")

    # Update data_config.py with best params 
    print(f"\n  To use the best params, update src/data_config.py ALS_PARAMS:")
    print(f"    rank     = {best_params['rank']}")
    print(f"    alpha    = {best_params['alpha']}")
    print(f"    regParam = {best_params['regParam']}")

    spark.stop()


if __name__ == "__main__":
    main()
