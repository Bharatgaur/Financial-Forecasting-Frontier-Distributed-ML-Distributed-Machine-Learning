# =============================================================================
# spark/feature_engineering.py  —  Feature Engineering using PySpark
# =============================================================================
# Run:  spark-submit spark/feature_engineering.py
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, MinMaxScaler, Imputer
)

# =============================================================================
# 1. SPARK SESSION
# =============================================================================
spark = (
    SparkSession.builder
    .appName("BankFeatureEngineering")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# =============================================================================
# 2. LOAD CLEANED DATA
# =============================================================================
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/bank.csv")
    .withColumnRenamed("default", "credit_default")
)
print(f"✓ Loaded {df.count()} rows")

# =============================================================================
# 3. HANDLE 'UNKNOWN' VALUES
# =============================================================================
# In this dataset, 'unknown' is the missing-value marker for categoricals.
# We replace with the column mode (most frequent value).

def get_mode(sdf, col_name):
    return (
        sdf.groupBy(col_name)
        .count()
        .orderBy(F.col("count").desc())
        .first()[col_name]
    )

cols_with_unknown = ["job", "education", "contact", "poutcome"]
for col in cols_with_unknown:
    mode_val = get_mode(df, col)
    df = df.withColumn(col,
        F.when(F.col(col) == "unknown", mode_val).otherwise(F.col(col))
    )
    print(f"  Replaced 'unknown' in '{col}' with mode: '{mode_val}'")

# =============================================================================
# 4. HANDLE pdays = -1  (client was never previously contacted)
# =============================================================================
# pdays = -1 means "no previous contact". We:
#   a) Create a boolean flag: was_contacted_before
#   b) Set pdays = 0 for those rows (so it's a valid integer)

df = (
    df
    .withColumn("was_contacted_before",
        F.when(F.col("pdays") == -1, 0).otherwise(1).cast("int"))
    .withColumn("pdays",
        F.when(F.col("pdays") == -1, 0).otherwise(F.col("pdays")))
)
print("✓ pdays processed, 'was_contacted_before' flag created")

# =============================================================================
# 5. FEATURE CREATION  (domain-driven new features)
# =============================================================================

# 5a. balance_per_age  → wealth-adjusted for age (proxy for savings rate)
df = df.withColumn("balance_per_age",
    F.round(F.col("balance") / F.col("age"), 2))

# 5b. is_high_balance  → customers with above-median balance are better targets
median_balance = df.approxQuantile("balance", [0.5], 0.01)[0]
df = df.withColumn("is_high_balance",
    F.when(F.col("balance") > median_balance, 1).otherwise(0).cast("int"))

# 5c. contact_intensity → campaign + previous combined effort score
df = df.withColumn("contact_intensity",
    F.col("campaign") + F.col("previous"))

# 5d. is_long_call  → calls > 5 minutes have much higher subscription rates
df = df.withColumn("is_long_call",
    F.when(F.col("duration") > 300, 1).otherwise(0).cast("int"))

# 5e. season  → map month to season (useful cyclic feature)
df = df.withColumn("season",
    F.when(F.col("month").isin("dec","jan","feb"), "winter")
     .when(F.col("month").isin("mar","apr","may"), "spring")
     .when(F.col("month").isin("jun","jul","aug"), "summer")
     .otherwise("autumn"))

# 5f. has_any_loan  → combined loan burden indicator
df = df.withColumn("has_any_loan",
    F.when((F.col("housing") == "yes") | (F.col("loan") == "yes"), 1)
     .otherwise(0).cast("int"))

print("✓ 6 new features created")
df.select("balance_per_age", "is_high_balance", "contact_intensity",
          "is_long_call", "season", "has_any_loan").show(3)

# =============================================================================
# 6. ENCODE TARGET VARIABLE
# =============================================================================
df = df.withColumn("label",
    F.when(F.col("y") == "yes", 1.0).otherwise(0.0))
print(f"✓ Target encoded  |  yes={df.filter(F.col('label')==1).count()}")

# =============================================================================
# 7. CATEGORICAL ENCODING PIPELINE
# =============================================================================
# Step A: StringIndexer  → converts string categories to integer indices
# Step B: OneHotEncoder  → converts indices to binary vectors (avoids ordinality)

categorical_cols = [
    "job", "marital", "education", "credit_default",
    "housing", "loan", "contact", "month", "poutcome", "season"
]

# StringIndexer: job → job_idx, marital → marital_idx, ...
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]

# OneHotEncoder: job_idx → job_ohe, ...
encoder = OneHotEncoder(
    inputCols  = [f"{c}_idx" for c in categorical_cols],
    outputCols = [f"{c}_ohe" for c in categorical_cols]
)

# =============================================================================
# 8. NUMERICAL FEATURES
# =============================================================================
numerical_cols = [
    "age", "balance", "duration", "campaign",
    "pdays", "previous", "day",
    "balance_per_age", "contact_intensity"
]

# =============================================================================
# 9. VECTOR ASSEMBLER
# =============================================================================
# Combines all features into a single 'features' vector column
# (required format for Spark ML algorithms)

all_feature_cols = (
    [f"{c}_ohe" for c in categorical_cols] +
    numerical_cols +
    ["was_contacted_before", "is_high_balance", "is_long_call", "has_any_loan"]
)

assembler = VectorAssembler(
    inputCols = all_feature_cols,
    outputCol = "features_raw",
    handleInvalid = "keep"
)

# =============================================================================
# 10. STANDARD SCALER
# =============================================================================
# Zero mean, unit variance — important for Logistic Regression & SVM
scaler = StandardScaler(
    inputCol  = "features_raw",
    outputCol = "features",
    withMean  = True,
    withStd   = True
)

# =============================================================================
# 11. BUILD & FIT PIPELINE
# =============================================================================
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler])

print("✓ Fitting feature pipeline…")
pipeline_model = pipeline.fit(df)
df_final = pipeline_model.transform(df)
print("✓ Pipeline fitted and applied")

# Preview
df_final.select("label", "features").show(5, truncate=True)
print(f"  Feature vector size: {df_final.select('features').first()[0].size}")

# =============================================================================
# 12. TRAIN / TEST SPLIT
# =============================================================================
# 80% train, 20% test — stratified not natively supported in PySpark,
# but random split is standard practice
train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

print(f"\n✓ Split complete")
print(f"  Train: {train_df.count()} rows")
print(f"  Test : {test_df.count()} rows")

# =============================================================================
# 13. SAVE PROCESSED DATASETS
# =============================================================================
train_df.select("label", "features").write.mode("overwrite").parquet(
    "data/train.parquet")
test_df.select("label", "features").write.mode("overwrite").parquet(
    "data/test.parquet")

# Also save the pipeline model for streaming inference later
pipeline_model.write().overwrite().save("data/feature_pipeline_model")

print("✓ train.parquet, test.parquet, and feature_pipeline_model saved")

spark.stop()
print("\n✓ FEATURE ENGINEERING COMPLETE")
