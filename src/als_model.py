# src/als_model.py
from pyspark.ml.recommendation import ALS
import src.data_config as config


def train_als(train_df):
    als = ALS(
        userCol          = config.USER_COL,
        itemCol          = config.ITEM_COL,
        ratingCol        = config.RATING_COL,
        coldStartStrategy= "drop",
        **config.ALS_PARAMS,
    )
    return als.fit(train_df)
