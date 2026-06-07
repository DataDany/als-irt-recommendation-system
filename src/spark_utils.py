# src/spark_utils.py
import os
import sys

# Pin both ends to the current interpreter before any Spark import touches the JVM.
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession  # noqa: E402  (import after env is set)


def get_spark(
    app_name:          str  = "MathRecommender",
    driver_memory:     str  = "8g",
    executor_memory:   str  = "8g",
    shuffle_partitions: int = 200,
    max_result_size:   str  = "2g",
    memory_fraction:   float = 0.6,
) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory",            driver_memory)
        .config("spark.executor.memory",           executor_memory)
        .config("spark.memory.fraction",           str(memory_fraction))
        .config("spark.sql.shuffle.partitions",    str(shuffle_partitions))
        .config("spark.driver.maxResultSize",      max_result_size)
        # Belt-and-suspenders: also pass through SparkConf so the JVM picks it up
        .config("spark.pyspark.python",            sys.executable)
        .config("spark.pyspark.driver.python",     sys.executable)
        .getOrCreate()
    )
