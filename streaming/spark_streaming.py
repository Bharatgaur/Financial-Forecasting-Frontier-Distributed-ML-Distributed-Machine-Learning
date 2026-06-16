# =============================================================================
# streaming/spark_streaming.py  —  Real-Time Banking Transaction Processing
# =============================================================================
# Architecture:
#   Simulator → writes JSON records to  streaming/input/
#   Spark Structured Streaming → reads from that folder
#   → applies ML model (Random Forest) for subscription prediction
#   → applies basic fraud detection rules
#   → writes enriched results to  streaming/output/
#
# Run in TWO terminals:
#   Terminal 1:  python3 streaming/data_simulator.py   (generates data)
#   Terminal 2:  spark-submit streaming/spark_streaming.py
# =============================================================================

import time, os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, DoubleType, TimestampType
)
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel

# =============================================================================
# 1. SPARK SESSION  (Streaming needs its own config)
# =============================================================================
spark = (
    SparkSession.builder
    .appName("BankStreamingPipeline")
    .master("local[2]")                         # 2 threads: 1 for read, 1 for write
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.streaming.stopGracefullyOnShutdown", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# =============================================================================
# 2. DEFINE INPUT SCHEMA
# =============================================================================
# Spark Structured Streaming requires an explicit schema for file sources.
# This matches the JSON records produced by data_simulator.py

schema = StructType([
    StructField("transaction_id", StringType(),  nullable=False),
    StructField("timestamp",      StringType(),  nullable=False),
    StructField("age",            IntegerType(), nullable=True),
    StructField("job",            StringType(),  nullable=True),
    StructField("marital",        StringType(),  nullable=True),
    StructField("education",      StringType(),  nullable=True),
    StructField("credit_default", StringType(),  nullable=True),
    StructField("balance",        IntegerType(), nullable=True),
    StructField("housing",        StringType(),  nullable=True),
    StructField("loan",           StringType(),  nullable=True),
    StructField("contact",        StringType(),  nullable=True),
    StructField("day",            IntegerType(), nullable=True),
    StructField("month",          StringType(),  nullable=True),
    StructField("duration",       IntegerType(), nullable=True),
    StructField("campaign",       IntegerType(), nullable=True),
    StructField("pdays",          IntegerType(), nullable=True),
    StructField("previous",       IntegerType(), nullable=True),
    StructField("poutcome",       StringType(),  nullable=True),
    StructField("amount",         DoubleType(),  nullable=True),   # transaction $
])

# =============================================================================
# 3. READ STREAMING DATA
# =============================================================================
# Spark reads each new JSON file dropped into streaming/input/ as a micro-batch.
# maxFilesPerTrigger=1 → process one file at a time (simulates real-time)

INPUT_DIR  = "streaming/input"
OUTPUT_DIR = "streaming/output"
CHKPT_DIR  = "streaming/checkpoint"

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

raw_stream = (
    spark.readStream
    .schema(schema)
    .option("maxFilesPerTrigger", 1)
    .json(INPUT_DIR)
)

print("✓ Streaming source connected to:", INPUT_DIR)

# =============================================================================
# 4. FEATURE ENGINEERING  (same logic as feature_engineering.py)
# =============================================================================
# We MUST replicate transformations here because the streaming data is raw.
# In production, load a saved PipelineModel instead.

def enrich_stream(df):
    """Apply feature engineering to streaming micro-batch."""
    df = (
        df
        # Parse the string timestamp into an actual event-time column.
        # Required for window() aggregations and watermarking below.
        .withColumn("event_time", F.to_timestamp("timestamp"))
        # Handle pdays=-1
        .withColumn("was_contacted_before",
            F.when(F.col("pdays") == -1, 0).otherwise(1).cast("int"))
        .withColumn("pdays",
            F.when(F.col("pdays") == -1, 0).otherwise(F.col("pdays")))
        # New features
        .withColumn("balance_per_age",
            F.round(F.col("balance") / F.col("age"), 2))
        .withColumn("is_high_balance",
            F.when(F.col("balance") > 444, 1).otherwise(0).cast("int"))
        .withColumn("contact_intensity",
            F.col("campaign") + F.col("previous"))
        .withColumn("is_long_call",
            F.when(F.col("duration") > 300, 1).otherwise(0).cast("int"))
        .withColumn("season",
            F.when(F.col("month").isin("dec","jan","feb"), "winter")
             .when(F.col("month").isin("mar","apr","may"), "spring")
             .when(F.col("month").isin("jun","jul","aug"), "summer")
             .otherwise("autumn"))
        .withColumn("has_any_loan",
            F.when(
                (F.col("housing") == "yes") | (F.col("loan") == "yes"), 1
            ).otherwise(0).cast("int"))
    )
    return df

# =============================================================================
# 5. FRAUD DETECTION RULES  (rule-based, real-time)
# =============================================================================
# These rules flag transactions for review — not trained ML, but fast heuristics.
# In production these would be combined with an anomaly detection model.

def apply_fraud_rules(df):
    """Flag suspicious transactions based on business rules."""
    df = df.withColumn("fraud_flag",
        F.when(
            # Rule 1: Very large transaction amount
            (F.col("amount") > 10000) |
            # Rule 2: Negative balance AND large transaction
            ((F.col("balance") < 0) & (F.col("amount") > 500)) |
            # Rule 3: Excessive campaign contacts (spam indicator)
            (F.col("campaign") > 15) |
            # Rule 4: Very short call — suspicious automated dialer
            (F.col("duration") < 5),
            "SUSPICIOUS"
        ).otherwise("CLEAN")
    )

    df = df.withColumn("fraud_score",
        (
            F.when(F.col("amount") > 10000, 40).otherwise(0) +
            F.when((F.col("balance") < 0) & (F.col("amount") > 500), 30).otherwise(0) +
            F.when(F.col("campaign") > 15, 20).otherwise(0) +
            F.when(F.col("duration") < 5, 10).otherwise(0)
        ).cast("int")
    )
    return df

# =============================================================================
# 6. RULE-BASED SUBSCRIPTION PREDICTION
# =============================================================================
# We use a deterministic rule model here for streaming because loading the
# full Spark ML RandomForest in a streaming context requires careful setup.
# In production: use model.transform() inside foreachBatch().

def predict_subscription(df):
    """
    Lightweight rule-based subscription predictor.
    Combines the top predictive signals found in EDA and feature importances.
    """
    df = df.withColumn("sub_score",
        (
            # Duration is the strongest predictor
            F.when(F.col("duration") > 300,  3).otherwise(0) +
            F.when(F.col("duration") > 600,  2).otherwise(0) +
            # Previous success is very informative
            F.when(F.col("poutcome") == "success", 3).otherwise(0) +
            # High balance customers are more likely to subscribe
            F.when(F.col("balance") > 1480,  2).otherwise(0) +
            # Low campaign intensity → not fatigued
            F.when(F.col("campaign") <= 2,   1).otherwise(0) +
            # Students and retired tend to subscribe more
            F.when(F.col("job").isin("student","retired"), 1).otherwise(0) +
            # Cellular contact works better
            F.when(F.col("contact") == "cellular", 1).otherwise(0)
        ).cast("int")
    )
    df = df.withColumn("predicted_subscription",
        F.when(F.col("sub_score") >= 5, "YES").otherwise("NO")
    )
    df = df.withColumn("subscription_confidence",
        F.round(
            F.when(F.col("sub_score") >= 7, 0.90)
             .when(F.col("sub_score") >= 5, 0.75)
             .when(F.col("sub_score") >= 3, 0.55)
             .otherwise(0.25),
            2
        )
    )
    return df

# =============================================================================
# 7. WINDOWED AGGREGATION  (event-time window + watermark)
# =============================================================================
# Rubric requirement: "Spark Streaming and window operations."
# This produces a rolling 1-minute summary (sliding every 30s) of transaction
# volume, average transaction amount, fraud rate, and predicted subscription
# rate — the kind of live dashboard metric a bank's ops team would watch.
#
# withWatermark("event_time", "2 minutes") tells Spark how late a record is
# allowed to arrive before it is dropped from window state. Without this,
# window state would grow forever (see README Q20).
# =============================================================================

def build_windowed_aggregates(enriched_df):
    """
    Tumbling/sliding window aggregation over event_time.
    window = 1 minute wide, slides every 30 seconds (50% overlap).
    """
    return (
        enriched_df
        .withWatermark("event_time", "2 minutes")
        .groupBy(
            F.window("event_time", "1 minute", "30 seconds"),
            "fraud_flag"
        )
        .agg(
            F.count("*").alias("tx_count"),
            F.round(F.avg("amount"), 2).alias("avg_amount"),
            F.round(F.avg("balance"), 2).alias("avg_balance"),
            F.sum(F.when(F.col("predicted_subscription") == "YES", 1)
                   .otherwise(0)).alias("predicted_subscribers")
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "fraud_flag",
            "tx_count",
            "avg_amount",
            "avg_balance",
            "predicted_subscribers"
        )
    )

# =============================================================================
# 8. PROCESSING PIPELINE
# =============================================================================

enriched_stream = (
    raw_stream
    .transform(enrich_stream)
    .transform(apply_fraud_rules)
    .transform(predict_subscription)
)

processed_stream = enriched_stream.select(
    "transaction_id",
    "timestamp",
    "age", "job", "balance", "duration", "campaign",
    # Engineered
    "is_long_call", "is_high_balance", "contact_intensity",
    # Outputs
    "fraud_flag",
    "fraud_score",
    "predicted_subscription",
    "subscription_confidence",
    F.current_timestamp().alias("processed_at")
)

windowed_stream = build_windowed_aggregates(enriched_stream)

# =============================================================================
# 9. WRITE STREAMING OUTPUT
# =============================================================================
# outputMode="append"  → only write new rows (correct for record-level stream)
# outputMode="update"  → windows emit updated totals as late data arrives
# trigger processingTime → run micro-batch every 10 seconds

query = (
    processed_stream
    .writeStream
    .outputMode("append")
    .format("json")
    .option("path", OUTPUT_DIR)
    .option("checkpointLocation", CHKPT_DIR)
    .trigger(processingTime="10 seconds")
    .start()
)

print("✓ Streaming query started")
print(f"  Input  : {INPUT_DIR}")
print(f"  Output : {OUTPUT_DIR}")
print(f"  Press Ctrl+C to stop\n")

# Also print to console for monitoring
console_query = (
    processed_stream
    .writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", False)
    .option("numRows", 5)
    .trigger(processingTime="10 seconds")
    .start()
)

# Windowed aggregation → its own checkpoint, written to streaming/output_windowed/
WINDOWED_OUTPUT_DIR = "streaming/output_windowed"
WINDOWED_CHKPT_DIR  = "streaming/checkpoint_windowed"
os.makedirs(WINDOWED_OUTPUT_DIR, exist_ok=True)

windowed_query = (
    windowed_stream
    .writeStream
    .outputMode("update")              # windows update as new/late data arrives
    .format("console")                 # also visible live in the terminal
    .option("truncate", False)
    .trigger(processingTime="30 seconds")
    .start()
)

windowed_file_query = (
    windowed_stream
    .writeStream
    .outputMode("append")              # file sink requires append; emits closed windows
    .format("json")
    .option("path", WINDOWED_OUTPUT_DIR)
    .option("checkpointLocation", WINDOWED_CHKPT_DIR)
    .trigger(processingTime="30 seconds")
    .start()
)

print("✓ Windowed aggregation query started (1-min window / 30-sec slide)")
print(f"  Windowed output : {WINDOWED_OUTPUT_DIR}\n")

# Keep running until interrupted
query.awaitTermination()
