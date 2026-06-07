# src/evaluation.py
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window
import src.data_config as config


def _explode_recs(recs: DataFrame, k: int) -> DataFrame:
    user_col = config.USER_COL
    item_col = config.ITEM_COL
    return (
        recs
        .select(user_col, F.posexplode("recommendations").alias("pos", "rec"))
        .select(
            F.col(user_col).cast("int").alias(user_col),
            F.col("pos").alias("pos"),
            F.col("rec")[item_col].cast("int").alias(item_col),
        )
        .withColumn("rank", F.col("pos") + F.lit(1))
        .filter(F.col("rank") <= k) 
        .drop("pos")
    )


def _test_positives(test_df: DataFrame) -> DataFrame:
    return (
        test_df
        .select(config.USER_COL, config.ITEM_COL)
        .dropDuplicates([config.USER_COL, config.ITEM_COL])
        .withColumn("_hit", F.lit(1))
    )


def hitrate_precision_recall_at_k(recs: DataFrame, test_df: DataFrame, k: int) -> dict:
    rec_e    = _explode_recs(recs, k)
    test_pos = _test_positives(test_df)
    hits     = rec_e.join(test_pos, on=[config.USER_COL, config.ITEM_COL], how="inner")
    n_users  = test_pos.select(config.USER_COL).distinct().count()
    if n_users == 0:
        return {"HR@K": 0.0, "Precision@K": 0.0, "Recall@K": 0.0}

    users_hit  = hits.select(config.USER_COL).distinct().count()
    total_hits = hits.count()
    total_test = test_pos.count()
    return {
        "HR@K":        float(users_hit / n_users),
        "Precision@K": float(total_hits / (n_users * k)),
        "Recall@K":    float(total_hits / total_test) if total_test > 0 else 0.0,
    }


def map_at_k(recs: DataFrame, test_df: DataFrame, k: int) -> float:
    rec_e    = _explode_recs(recs, k)
    test_pos = _test_positives(test_df) 
    n_users  = test_pos.select(config.USER_COL).distinct().count()
    if n_users == 0:
        return 0.0

    win = Window.partitionBy(config.USER_COL).orderBy("rank").rowsBetween(Window.unboundedPreceding, 0)

    rel = (
        rec_e
        .join(test_pos, on=[config.USER_COL, config.ITEM_COL], how="left")
        .withColumn("is_rel", F.when(F.col("_hit").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
    )

    rel2 = (
        rel
        .withColumn("cum_rel", F.sum("is_rel").over(win))
        .withColumn("prec_at_rank", F.col("cum_rel") / F.col("rank"))
        .where(F.col("is_rel") == 1)
    )

    test_cnt = test_pos.groupBy(config.USER_COL).agg(F.count("*").alias("n_test"))

    ap = (
        rel2.groupBy(config.USER_COL).agg(F.avg("prec_at_rank").alias("ap"))
        .join(test_cnt, on=config.USER_COL, how="inner")
    )

    ap_full = (
        test_pos.select(config.USER_COL).distinct()
        .join(ap.select(config.USER_COL, "ap"), on=config.USER_COL, how="left")
        .na.fill({"ap": 0.0})
    )
    return float(ap_full.agg(F.avg("ap")).collect()[0][0])


def evaluate_all_k(recs: DataFrame, test_df: DataFrame, k: int,
                   compute_map_ndcg: bool = True) -> dict:
    """
    Convenience wrapper: compute all metrics for a given K in one call.
    Popularity / Random baselines skip MAP and NDCG (pass compute_map_ndcg=False).
    """
    import src.data_config as config
    m = hitrate_precision_recall_at_k(recs, test_df, k)
    if compute_map_ndcg:
        m["MAP@K"]  = map_at_k(recs, test_df, k)
        m["NDCG@K"] = ndcg_at_k(recs, test_df, k)
    return m


def ndcg_at_k(recs: DataFrame, test_df: DataFrame, k: int) -> float:
    rec_e    = _explode_recs(recs, k)
    test_pos = _test_positives(test_df)
    n_users  = test_pos.select(config.USER_COL).distinct().count()
    if n_users == 0:
        return 0.0

    rel = (
        rec_e
        .join(test_pos, on=[config.USER_COL, config.ITEM_COL], how="left")
        .withColumn("rel",      F.when(F.col("_hit").isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("discount", F.lit(1.0) / F.log2(F.col("rank") + F.lit(1.0)))
        .withColumn("dcg_term", F.col("rel") * F.col("discount"))
    )

    dcg = rel.groupBy(config.USER_COL).agg(F.sum("dcg_term").alias("dcg"))

    test_cnt = test_pos.groupBy(config.USER_COL).agg(F.count("*").alias("n_test"))
    idcg = (
        test_cnt
        .withColumn("m", F.least(F.col("n_test"), F.lit(k)).cast("int"))
        .withColumn("ranks", F.expr("sequence(1, m)"))
        .select(config.USER_COL, F.explode("ranks").alias("rank"))
        .withColumn("discount", F.lit(1.0) / F.log2(F.col("rank") + F.lit(1.0)))
        .groupBy(config.USER_COL).agg(F.sum("discount").alias("idcg"))
    )

    result = (
        test_pos.select(config.USER_COL).distinct()
        .join(dcg,  on=config.USER_COL, how="left")
        .join(idcg, on=config.USER_COL, how="left")
        .na.fill({"dcg": 0.0, "idcg": 0.0})
        .withColumn("ndcg", F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(F.lit(0.0)))
        .agg(F.avg("ndcg"))
        .collect()[0][0]
    )
    return float(result)
