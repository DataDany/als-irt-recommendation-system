# src/data_loader.py
from src.data_config import DATA_PATH, SEED


def load_data(spark):
    return spark.read.parquet(DATA_PATH)


def split_data(df):
    return df.randomSplit([0.8, 0.2], seed=SEED)
