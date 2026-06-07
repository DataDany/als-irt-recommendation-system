# src/baselines.py
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window
import src.data_config as config


def popularity_recs(train_df: DataFrame, k: int) -> DataFrame:
    """
    Top-K globally most interacted items recommended to every user.
    """
    # 1. rank items by total interaction weight, keep top-k
    top_items = (
        train_df
        .groupBy(config.ITEM_COL)
        .agg(F.sum(config.RATING_COL).alias("score"))
        .orderBy(F.desc("score"))
        .limit(k)
    )

    # 2. collapse top-k into a single list (one row)
    pop_list = top_items.agg(
        F.collect_list(
            F.struct(F.col(config.ITEM_COL), F.col("score"))
        ).alias("recommendations")
    )

    # 3. broadcast that list to every user
    users = train_df.select(config.USER_COL).distinct()
    return users.crossJoin(pop_list)


def random_recs(train_df: DataFrame, k: int, seed: int = config.SEED) -> DataFrame:
    """
    K random items per user (baseline).
    """
    # Sample k distinct items globally
    all_items = train_df.select(config.ITEM_COL).distinct()
    rand_items = (
        all_items
        .orderBy(F.rand(seed))
        .limit(k)
    )

    # Collapse into a single list row, then broadcast to every user
    rand_list = rand_items.agg(
        F.collect_list(
            F.struct(F.col(config.ITEM_COL), F.lit(1.0).alias("score"))
        ).alias("recommendations")
    )

    users = train_df.select(config.USER_COL).distinct()
    return users.crossJoin(rand_list)
