# =============================================================================
# spark/eda.py  —  Exploratory Data Analysis using PySpark
# =============================================================================
# Run:  spark-submit spark/eda.py
# Or:   python3 spark/eda.py   (uses local Spark session)
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
import os

# ── Optional: matplotlib/seaborn for local visualization ─────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend (no display needed)
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    PLOT = True
except ImportError:
    PLOT = False
    print("[INFO] matplotlib/seaborn not found. Skipping plots.")

# =============================================================================
# 1. SPARK SESSION
# =============================================================================
spark = (
    SparkSession.builder
    .appName("BankEDA")
    # Run locally with all available CPU cores
    .master("local[*]")
    # Tune shuffle partitions for a small dataset (4521 rows)
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("✓ SparkSession created")

# =============================================================================
# 2. LOAD DATA
# =============================================================================
# We load from local CSV so the script works without a live Hadoop cluster.
# In production, replace the path with: hdfs:///user/bankproject/raw/bank/bank.csv
DATA_PATH = "data/bank.csv"

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(DATA_PATH)
)

# 'default' is a reserved keyword in Spark SQL — rename it
df = df.withColumnRenamed("default", "credit_default")

print(f"✓ Data loaded: {df.count()} rows × {len(df.columns)} columns")

# =============================================================================
# 3. SCHEMA & BASIC OVERVIEW
# =============================================================================
print("\n" + "="*60)
print("SCHEMA")
print("="*60)
df.printSchema()

print("\n" + "="*60)
print("FIRST 5 ROWS")
print("="*60)
df.show(5, truncate=False)

# =============================================================================
# 4. DATA QUALITY CHECK
# =============================================================================
print("\n" + "="*60)
print("NULL / MISSING VALUE ANALYSIS")
print("="*60)

# Count nulls in every column
null_counts = df.select([
    F.count(F.when(F.col(c).isNull() | (F.col(c) == ""), c)).alias(c)
    for c in df.columns
])
null_counts.show(truncate=False)

# Count 'unknown' values (domain-specific missing marker)
print("\n--- 'unknown' value counts per column ---")
for col in ["job", "education", "contact", "poutcome"]:
    cnt = df.filter(F.col(col) == "unknown").count()
    print(f"  {col:15s}: {cnt}")

# Duplicate check
dup_count = df.count() - df.dropDuplicates().count()
print(f"\n--- Duplicate rows: {dup_count} ---")

# =============================================================================
# 5. TARGET VARIABLE DISTRIBUTION
# =============================================================================
print("\n" + "="*60)
print("TARGET VARIABLE  (y = subscribed?)")
print("="*60)

target_dist = (
    df.groupBy("y")
    .agg(F.count("*").alias("count"))
    .withColumn("percentage",
        F.round(F.col("count") / df.count() * 100, 2))
    .orderBy("y")
)
target_dist.show()

# Class imbalance ratio
yes_count = df.filter(F.col("y") == "yes").count()
no_count  = df.filter(F.col("y") == "no").count()
print(f"  Class imbalance ratio  no:yes = {no_count}:{yes_count}"
      f"  ({no_count/yes_count:.1f}x imbalance)")

# =============================================================================
# 6. NUMERICAL FEATURE STATISTICS
# =============================================================================
print("\n" + "="*60)
print("NUMERICAL FEATURE STATISTICS")
print("="*60)

num_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
df.select(num_cols).describe().show()

# Outlier detection via IQR (shown for balance)
q1, q3 = df.approxQuantile("balance", [0.25, 0.75], 0.01)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = df.filter(
    (F.col("balance") < lower_fence) | (F.col("balance") > upper_fence)
).count()
print(f"\n  Balance IQR range: [{lower_fence:.0f}, {upper_fence:.0f}]")
print(f"  Outlier rows (balance): {outliers} ({outliers/df.count()*100:.1f}%)")

# =============================================================================
# 7. CATEGORICAL FEATURE ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("CATEGORICAL FEATURE ANALYSIS")
print("="*60)

cat_cols = ["job", "marital", "education", "credit_default",
            "housing", "loan", "contact", "month", "poutcome"]

for col in cat_cols:
    print(f"\n--- {col} ---")
    (
        df.groupBy(col)
        .agg(
            F.count("*").alias("count"),
            F.round(F.count("*") / df.count() * 100, 2).alias("pct")
        )
        .orderBy(F.col("count").desc())
    ).show(truncate=False)

# =============================================================================
# 8. SUBSCRIPTION RATE BY FEATURE  (key insight extraction)
# =============================================================================
print("\n" + "="*60)
print("SUBSCRIPTION RATE BY KEY FEATURES")
print("="*60)

def sub_rate(group_col):
    return (
        df.groupBy(group_col)
        .agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("y") == "yes", 1).otherwise(0)).alias("subscribed")
        )
        .withColumn("sub_rate_pct",
            F.round(F.col("subscribed") / F.col("total") * 100, 2))
        .orderBy(F.col("sub_rate_pct").desc())
    )

for col in ["job", "education", "marital", "poutcome", "month"]:
    print(f"\n--- Subscription rate by {col} ---")
    sub_rate(col).show(truncate=False)

# =============================================================================
# 9. CORRELATION ANALYSIS (numerical features)
# =============================================================================
print("\n" + "="*60)
print("PAIRWISE CORRELATIONS WITH BALANCE")
print("="*60)

for col in ["age", "duration", "campaign", "previous"]:
    corr = df.stat.corr("balance", col)
    print(f"  balance ↔ {col:12s}: {corr:.4f}")

# =============================================================================
# 10. VISUALIZATIONS  (saved as PNG files)
# =============================================================================
if PLOT:
    os.makedirs("docs/plots", exist_ok=True)
    pdf = df.toPandas()
    sns.set_style("whitegrid")

    # -- Plot 1: Target distribution -----------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    pdf["y"].value_counts().plot(kind="bar", color=["#e74c3c", "#2ecc71"], ax=ax)
    ax.set_title("Subscription Distribution (Target: y)")
    ax.set_xlabel("Subscribed?")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["No", "Yes"], rotation=0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig("docs/plots/01_target_distribution.png", dpi=150)
    plt.close()

    # -- Plot 2: Subscription rate by job ------------------------------------
    job_df = (pdf.groupby("job")["y"]
               .apply(lambda x: (x == "yes").mean() * 100)
               .reset_index(name="sub_rate")
               .sort_values("sub_rate", ascending=True))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(job_df["job"], job_df["sub_rate"], color="#3498db")
    ax.set_title("Subscription Rate by Job Type (%)")
    ax.set_xlabel("Subscription Rate (%)")
    plt.tight_layout()
    plt.savefig("docs/plots/02_sub_rate_by_job.png", dpi=150)
    plt.close()

    # -- Plot 3: Age distribution by subscription ----------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [("yes", "#2ecc71"), ("no", "#e74c3c")]:
        pdf[pdf["y"] == label]["age"].plot(
            kind="hist", bins=20, alpha=0.6, label=label, color=color, ax=ax)
    ax.set_title("Age Distribution by Subscription")
    ax.set_xlabel("Age")
    ax.legend()
    plt.tight_layout()
    plt.savefig("docs/plots/03_age_distribution.png", dpi=150)
    plt.close()

    # -- Plot 4: Call duration by subscription --------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    pdf.boxplot(column="duration", by="y", ax=ax, patch_artist=True)
    plt.suptitle("")
    ax.set_title("Call Duration by Subscription Outcome")
    ax.set_xlabel("Subscribed?")
    ax.set_ylabel("Duration (seconds)")
    plt.tight_layout()
    plt.savefig("docs/plots/04_duration_by_subscription.png", dpi=150)
    plt.close()

    # -- Plot 5: Subscription rate by month ----------------------------------
    month_order = ["jan","feb","mar","apr","may","jun",
                   "jul","aug","sep","oct","nov","dec"]
    month_df = (pdf.groupby("month")["y"]
                  .apply(lambda x: (x == "yes").mean() * 100)
                  .reindex(month_order).dropna()
                  .reset_index(name="sub_rate"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(month_df["month"], month_df["sub_rate"], color="#9b59b6")
    ax.set_title("Subscription Rate by Month (%)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Subscription Rate (%)")
    plt.tight_layout()
    plt.savefig("docs/plots/05_sub_rate_by_month.png", dpi=150)
    plt.close()

    print("✓ All 5 plots saved to docs/plots/")

# =============================================================================
# 11. SAVE CLEAN DATA BACK TO HDFS (parquet for downstream Spark jobs)
# =============================================================================
out_path = "data/bank_clean.parquet"
(
    df
    .dropDuplicates()
    .write
    .mode("overwrite")
    .parquet(out_path)
)
print(f"\n✓ Clean data saved → {out_path}")

spark.stop()
print("\n✓ EDA COMPLETE")
