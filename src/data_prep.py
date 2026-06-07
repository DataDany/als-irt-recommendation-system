# src/data_prep.py
from pyspark.sql import DataFrame, functions as F
import src.data_config as config

_BASE_TIME_SEC = 35.0
_TIME_PER_DIFF = 18.0


def prepare_for_als(interactions: DataFrame) -> DataFrame:
    """
    Returns (user_id, task_id, rating) ready for ALS.

    RATING_MODE = "quality" - weighted score from correctness / attempts / hints / time
    RATING_MODE = "binary" - every interaction = 1.0 (original implicit behaviour)
    """
    user_col = config.USER_COL
    item_col = config.ITEM_COL

    df = (interactions
          .withColumn(user_col, F.col(user_col).cast("int"))
          .withColumn(item_col, F.col(item_col).cast("int")))

    if config.RATING_COL in df.columns:
        df = df.withColumn(config.RATING_COL, F.col(config.RATING_COL).cast("float"))
    elif config.RATING_MODE == "correct_count" and "is_correct" in df.columns:
        df = df.withColumn("rating", F.col("is_correct").cast("float"))
    elif config.RATING_MODE == "log_correct" and "is_correct" in df.columns:
        df = df.withColumn("rating", F.col("is_correct").cast("float"))
    elif config.RATING_MODE == "quality" and "is_correct" in df.columns:
        df = _add_quality_rating(df)
    elif config.RATING_MODE == "binary":
        df = df.withColumn("rating", F.lit(1.0).cast("float"))
    else:
        df = df.withColumn("rating", F.lit(1.0).cast("float"))

    df = df.select(user_col, item_col, config.RATING_COL).na.drop()
    df = df.groupBy(user_col, item_col).agg(F.sum(config.RATING_COL).alias(config.RATING_COL))

    if config.RATING_MODE == "log_correct":
        df = df.withColumn(
            config.RATING_COL,
            F.log1p(F.col(config.RATING_COL)).cast("float"),
        )

    return df


def _add_quality_rating(df: DataFrame) -> DataFrame:
    """
    Quality-weighted implicit rating (always positive for ALS implicit feedback).

    base        = 1.0 if correct, 0.35 if incorrect
    attempt_pen = 0.12 × (attempts – 1)
    hint_pen    = 0.10 × hints_used
    time_bonus  = 0.08 × max(0, 1 – actual/expected) × is_correct
    rating      = max(0.10, base – penalties + bonus)
    """
    base           = F.when(F.col("is_correct") == 1, F.lit(1.0)).otherwise(F.lit(0.35))
    attempt_penalty = F.lit(0.12) * (F.col("attempts").cast("float") - F.lit(1.0))
    hint_penalty    = F.lit(0.10) * F.col("hints_used").cast("float")

    expected_time = F.lit(_BASE_TIME_SEC) + F.lit(_TIME_PER_DIFF) * F.col("difficulty")
    time_ratio    = F.col("time_spent_sec") / expected_time
    time_bonus    = (
        F.lit(0.08)
        * F.when(time_ratio < F.lit(1.0), F.lit(1.0) - time_ratio).otherwise(F.lit(0.0))
        * F.col("is_correct").cast("float")
    )

    quality = F.greatest(base - attempt_penalty - hint_penalty + time_bonus, F.lit(0.10)).cast("float")
    return df.withColumn("rating", quality)
