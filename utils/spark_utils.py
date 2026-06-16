# =============================================================================
# utils/spark_utils.py  —  Shared Utility Functions
# =============================================================================

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
import json, os, time

# =============================================================================
# SESSION FACTORY
# =============================================================================

def get_spark_session(app_name: str, local: bool = True,
                      memory: str = "2g") -> SparkSession:
    """
    Create or reuse a SparkSession.

    Parameters
    ----------
    app_name : str   Name shown in Spark UI
    local    : bool  True → local[*], False → use cluster YARN/standalone
    memory   : str   Driver memory (e.g. '2g', '4g')
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", memory)
    )
    if local:
        builder = builder.master("local[*]")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv(spark: SparkSession, path: str,
             rename_default: bool = True) -> DataFrame:
    """
    Load a CSV file with header and schema inference.
    'default' is a Spark SQL keyword — rename_default replaces it with
    'credit_default' to avoid parser errors.
    """
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )
    if rename_default and "default" in df.columns:
        df = df.withColumnRenamed("default", "credit_default")
    return df


# =============================================================================
# DATA QUALITY
# =============================================================================

def null_report(df: DataFrame) -> None:
    """Print null and 'unknown' counts for every column."""
    print("\n--- NULL REPORT ---")
    null_row = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()
    for col, cnt in null_row.items():
        flag = " ⚠" if cnt > 0 else ""
        print(f"  {col:20s}: {cnt} nulls{flag}")

    print("\n--- 'unknown' COUNTS ---")
    str_cols = [f.name for f in df.schema.fields
                if str(f.dataType) == "StringType()"]
    for col in str_cols:
        cnt = df.filter(F.col(col) == "unknown").count()
        if cnt > 0:
            print(f"  {col:20s}: {cnt}")


def class_balance(df: DataFrame, label_col: str = "y") -> None:
    """Print class distribution and imbalance ratio."""
    dist = (
        df.groupBy(label_col)
        .count()
        .withColumn("pct", F.round(F.col("count") / df.count() * 100, 2))
        .orderBy(label_col)
    )
    print(f"\n--- CLASS BALANCE ({label_col}) ---")
    dist.show()
    counts = {row[label_col]: row["count"] for row in dist.collect()}
    if "yes" in counts and "no" in counts:
        ratio = counts["no"] / counts["yes"]
        print(f"  Imbalance ratio (no:yes) = {ratio:.1f}x")


# =============================================================================
# PARTITIONING UTILITIES
# =============================================================================

def repartition_for_ml(df: DataFrame, num_partitions: int = 8) -> DataFrame:
    """
    Repartition DataFrame for ML training.
    More partitions → better parallelism but higher overhead.
    Rule of thumb: 2–4× the number of available CPU cores.
    """
    current = df.rdd.getNumPartitions()
    print(f"  Repartitioning: {current} → {num_partitions} partitions")
    return df.repartition(num_partitions)


def show_partition_info(df: DataFrame, label: str = "") -> None:
    """Show partition count and per-partition row distribution."""
    n = df.rdd.getNumPartitions()
    dist = df.rdd.mapPartitionsWithIndex(
        lambda i, rows: [(i, sum(1 for _ in rows))]
    ).collect()
    print(f"\n--- PARTITION INFO  {label} ---")
    print(f"  Total partitions: {n}")
    for pid, cnt in sorted(dist):
        bar = "█" * (cnt // 50)
        print(f"  Part {pid:2d}: {cnt:5d} rows  {bar}")


# =============================================================================
# HDFS UTILITIES  (requires pyspark with Hadoop configured)
# =============================================================================

def save_to_hdfs(df: DataFrame, hdfs_path: str,
                 fmt: str = "parquet", mode: str = "overwrite") -> None:
    """Save a DataFrame to HDFS."""
    start = time.time()
    df.write.mode(mode).format(fmt).save(hdfs_path)
    elapsed = time.time() - start
    print(f"✓ Saved to HDFS: {hdfs_path}  [{elapsed:.1f}s]")


def read_from_hdfs(spark: SparkSession, hdfs_path: str,
                   fmt: str = "parquet") -> DataFrame:
    """Read a DataFrame from HDFS."""
    return spark.read.format(fmt).load(hdfs_path)


# =============================================================================
# METRICS REPORTING
# =============================================================================

def save_metrics(metrics: dict, path: str = "docs/metrics.json") -> None:
    """Append model metrics to a JSON report file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.append(metrics)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"✓ Metrics saved → {path}")
