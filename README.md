# 🏦 Capstone Project 2: Distributed Machine Learning — Financial Forecasting Frontier

> **Technologies:** Apache Hadoop · Apache Hive · Apache Spark (PySpark) · Spark Streaming · Spark ML  
> **Domain:** Banking & Financial Services  
> **Dataset:** `bank.csv` — 4,521 customer records, 17 features  
> **Target:** Predict whether a customer will subscribe to a term deposit (`y`)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Use Cases](#2-business-use-cases)
3. [System Architecture](#3-system-architecture)
4. [Dataset Description](#4-dataset-description)
5. [HDFS Storage Setup](#5-hdfs-storage-setup)
6. [Hive Analytics Layer](#6-hive-analytics-layer)
7. [EDA with PySpark](#7-eda-with-pyspark)
8. [Feature Engineering](#8-feature-engineering)
9. [Machine Learning Results](#9-machine-learning-results)
10. [Real-Time Streaming Pipeline](#10-real-time-streaming-pipeline)
11. [Data Parallelism & Optimization](#11-data-parallelism--optimization)
12. [Project Folder Structure](#12-project-folder-structure)
13. [Setup & Execution Guide](#13-setup--execution-guide)
14. [Interview & Viva Questions](#14-interview--viva-questions)
15. [Common Errors & Solutions](#15-common-errors--solutions)

---

## 1. Project Overview

Modern banks run thousands of phone-based marketing campaigns each year to sell products like term deposits. The challenge is identifying **which customers are likely to say yes** — before making the call. Calling the wrong customers wastes agent time and annoys customers.

This project demonstrates a **complete big data + machine learning pipeline** that:
- Stores raw data efficiently on **HDFS** (Hadoop Distributed File System)
- Queries it at scale using **Hive SQL**
- Performs exploratory analysis with **Apache Spark**
- Engineers features and trains **three ML models** using Spark ML
- Serves **real-time predictions** via Spark Structured Streaming

The same architecture used here is deployed at banks like HDFC, ICICI, and JPMorgan Chase to drive their data platforms.

---

## 2. Business Use Cases

### 2a. Customer Subscription Prediction *(primary goal)*
Predict whether a customer will subscribe to a term deposit based on their demographics, account information, and past campaign interaction.

**Business value:** A bank with 1 million customers can reduce call volume by 60% while maintaining 90% of subscriptions — saving crores in call-centre costs.

### 2b. Loan/Default Risk Analysis
Using `balance`, `housing`, `loan`, and `credit_default` fields, segment customers by financial stress level and flag high-risk profiles before offering credit products.

### 2c. Marketing Campaign Optimization
Analyze which months, call durations, and contact methods yield the highest conversion. The data shows that:
- **October** has a 46.25% subscription rate vs **May** at 6.65%
- Calls longer than **5 minutes** convert at 33–56% vs 3% for short calls
- Previous campaign **successes** are the strongest predictor of future subscription

### 2d. Customer Segmentation
Group customers into meaningful behavioral clusters (high-balance seniors, loan-burdened young professionals, students) for targeted product offerings.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                             │
│   bank.csv (local)  →  hdfs dfs -put  →  HDFS /user/bankproject/   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    STORAGE LAYER  (HDFS)                            │
│  /user/bankproject/                                                 │
│  ├── raw/bank/bank.csv          ← Original CSV                      │
│  ├── processed/features/        ← Parquet after EDA                 │
│  ├── models/random_forest/      ← Saved Spark ML model              │
│  └── streaming/input|output/    ← Real-time records                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
┌─────────▼──────────┐               ┌──────────▼──────────┐
│   QUERY LAYER      │               │  PROCESSING LAYER   │
│   Apache Hive      │               │   Apache Spark      │
│                    │               │                     │
│  • CREATE TABLE    │               │  eda.py             │
│  • ORC storage     │               │  feature_eng.py     │
│  • SQL analytics   │               │  model_training.py  │
│  • Segmentation    │               │  Spark ML Pipeline  │
└────────────────────┘               └──────────┬──────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │  STREAMING LAYER    │
                                     │  Spark Structured   │
                                     │  Streaming          │
                                     │                     │
                                     │  data_simulator.py  │
                                     │  → JSON files       │
                                     │  → fraud detection  │
                                     │  → ML prediction    │
                                     └─────────────────────┘
```

### Data Flow (End-to-End)

```
[bank.csv]
    ↓ hdfs dfs -put
[HDFS: raw/bank/bank.csv]
    ↓ Hive EXTERNAL TABLE
[HiveQL queries → insights]
    ↓ PySpark spark.read.csv
[DataFrame: 4521 rows × 17 cols]
    ↓ spark/eda.py
[Cleaned DataFrame + plots]
    ↓ spark/feature_engineering.py
[Feature vectors: 50+ dimensions]
    ↓ spark/model_training.py
[Trained Models: LR, DT, RF]
    ↓ best model saved
[data/models/random_forest/]
    ↓ streaming/spark_streaming.py
[Real-time predictions + fraud flags]
```

---

## 4. Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Customer age (19–87) |
| `job` | string | 12 job categories (management, blue-collar, etc.) |
| `marital` | string | married / single / divorced |
| `education` | string | primary / secondary / tertiary |
| `default` | string | Credit default: yes / no |
| `balance` | int | Avg yearly balance in euros (-3313 to 71188) |
| `housing` | string | Has housing loan: yes / no |
| `loan` | string | Has personal loan: yes / no |
| `contact` | string | cellular / telephone / unknown |
| `day` | int | Last contact day of month |
| `month` | string | Last contact month |
| `duration` | int | Call duration in seconds (key predictor!) |
| `campaign` | int | Contacts in this campaign |
| `pdays` | int | Days since last contacted (-1 = never) |
| `previous` | int | Contacts before this campaign |
| `poutcome` | string | Previous campaign outcome |
| **`y`** | string | **TARGET: subscribed? yes / no** |

**Key statistics:**
- Total records: **4,521**
- Class imbalance: **88.5% No, 11.5% Yes** (7.7× imbalance)
- No null values (but 'unknown' used as missing marker)

---

## 5. HDFS Storage Setup

### What is HDFS?

HDFS (Hadoop Distributed File System) splits large files into **128 MB blocks** and distributes them across multiple DataNodes. Each block is replicated 3× for fault tolerance. A NameNode maintains the directory tree and block locations.

```
NameNode (metadata only)
     │
     ├── DataNode 1  [block_001, block_004, ...]
     ├── DataNode 2  [block_002, block_001, ...]   ← replica
     └── DataNode 3  [block_003, block_002, ...]   ← replica
```

### HDFS Commands

```bash
# Start Hadoop services
start-dfs.sh
start-yarn.sh

# Create project directory tree
hdfs dfs -mkdir -p /user/bankproject/raw/bank
hdfs dfs -mkdir -p /user/bankproject/processed/features
hdfs dfs -mkdir -p /user/bankproject/models/random_forest
hdfs dfs -mkdir -p /user/bankproject/streaming/input
hdfs dfs -mkdir -p /user/bankproject/streaming/output

# Upload dataset
hdfs dfs -put data/bank.csv /user/bankproject/raw/bank/

# Verify
hdfs dfs -ls /user/bankproject/raw/bank/
hdfs dfs -du -h /user/bankproject/

# Download a file back to local
hdfs dfs -get /user/bankproject/raw/bank/bank.csv /tmp/

# View file content (first 10 lines)
hdfs dfs -cat /user/bankproject/raw/bank/bank.csv | head -10

# Delete a directory
hdfs dfs -rm -r /user/bankproject/processed/old/

# Check replication
hdfs fsck /user/bankproject/raw/bank/bank.csv -files -blocks
```

### Full HDFS Folder Structure

```
hdfs://namenode:9000/user/bankproject/
├── raw/
│   └── bank/
│       └── bank.csv                ← 360 KB, 4521 rows
├── processed/
│   ├── features/
│   │   ├── train.parquet/          ← 3662 rows (80%)
│   │   └── test.parquet/           ← 859 rows (20%)
│   └── encoded/
│       └── feature_pipeline_model/ ← Serialized ML Pipeline
├── models/
│   ├── logistic_regression/        ← Saved LR model
│   ├── decision_tree/              ← Saved DT model
│   └── random_forest/              ← Best model (AUC=0.9119)
├── hive/
│   └── warehouse/
│       ├── bankdb.db/bank_raw/     ← External Hive table
│       └── bankdb.db/bank_orc/     ← ORC-compressed table
├── streaming/
│   ├── input/                      ← Drop JSON files here
│   ├── output/                     ← Enriched predictions land here
│   └── checkpoint/                 ← Spark Streaming state
└── reports/
    └── model_results.json
```

---

## 6. Hive Analytics Layer

### What is Hive?

Apache Hive provides a SQL interface on top of HDFS. It compiles HiveQL queries into MapReduce or Tez jobs that run in parallel across the cluster.

**Schema Design Decisions:**
- Use `EXTERNAL TABLE` so HDFS data is not deleted if table is dropped
- Store analytics table as **ORC + Snappy** for 60–70% compression
- Column naming: `default` → `def` (reserved SQL keyword)

### Running Hive Queries

```bash
# Start Hive shell
hive

# Run script file
hive -f hive/bank_hive.sql

# Or from beeline (production)
beeline -u jdbc:hive2://localhost:10000 -f hive/bank_hive.sql
```

### Key Query Results

**Subscription rate by job:**
| Job | Sub Rate |
|-----|----------|
| retired | 23.5% |
| student | 22.6% |
| management | 13.5% |
| blue-collar | 7.3% |

**Subscription rate by month:**
| Month | Rate |
|-------|------|
| October | 46.3% |
| December | 45.0% |
| March | 42.9% |
| May | 6.7% |

---

## 7. EDA with PySpark

```bash
spark-submit spark/eda.py
```

### Key Findings

| Insight | Finding |
|---------|---------|
| Class imbalance | 88.5% No vs 11.5% Yes |
| Strongest predictor | Call duration (>5 min → 33%+ conversion) |
| Best job segment | Retired (23.5%) & Students (22.6%) |
| Best months | Oct, Dec, Mar (30–46%) vs May (6.7%) |
| Previous success | 65% re-subscribe if previous outcome = success |
| Balance effect | High-balance customers subscribe 2× more |

### Correlation Summary

| Feature pair | Correlation |
|---|---|
| balance ↔ age | 0.098 |
| balance ↔ duration | 0.014 |
| balance ↔ campaign | -0.052 |
| balance ↔ previous | 0.021 |

---

## 8. Feature Engineering

```bash
spark-submit spark/feature_engineering.py
```

### Engineered Features

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `was_contacted_before` | pdays != -1 | Prior contact = warm lead |
| `is_long_call` | duration > 300s | Strong subscription signal |
| `is_high_balance` | balance > median | Wealthier = better prospect |
| `balance_per_age` | balance / age | Age-adjusted savings proxy |
| `contact_intensity` | campaign + previous | Total effort spent |
| `has_any_loan` | housing='yes' OR loan='yes' | Debt burden flag |
| `season` | month → season | Cyclical campaign patterns |

### Pipeline Architecture

```
Raw CSV
  │
  ├── StringIndexer   (job → job_idx, marital → marital_idx, ...)
  ├── OneHotEncoder   (job_idx → job_ohe, ...)
  ├── VectorAssembler (all features → features_raw vector)
  └── StandardScaler  (features_raw → features, mean=0, std=1)
                               │
                          [Label: 0.0 / 1.0]
                          [Features: 50-dim vector]
                               │
                    ┌──────────┴──────────┐
                 train_df (80%)      test_df (20%)
                  3,662 rows          859 rows
```

---

## 9. Machine Learning Results

```bash
spark-submit spark/model_training.py
```

### Real Model Performance (trained on this dataset)

| Model | Accuracy | F1-Score | Recall | ROC-AUC |
|-------|----------|----------|--------|---------|
| Logistic Regression | **89.52%** | 0.8774 | 0.8952 | 0.8868 |
| Decision Tree (depth=8) | 89.17% | 0.8841 | 0.8917 | 0.3403 |
| **Random Forest (100 trees)** | 89.41% | 0.8656 | 0.8941 | **0.9119** ✓ |

### 🏆 Winner: Random Forest
- **AUC = 0.9119** — best discrimination between subscribers and non-subscribers
- Decision Tree has poor AUC (0.34) because it overfits without ensemble averaging
- Logistic Regression is a strong baseline (AUC=0.89) and very interpretable

### Threshold Analysis

Default threshold = 0.5 (maximize precision).  
At threshold = 0.3 (maximize recall → catch more potential subscribers):
- Recall improves significantly — fewer missed opportunities
- Suitable when the cost of missing a subscriber > cost of an extra call

### Hyperparameter Tuning (Random Forest)

Grid searched: `numTrees ∈ {50, 100}` × `maxDepth ∈ {5, 10}`  
3-fold cross-validation, metric = AUC  
**Best:** `numTrees=100, maxDepth=10`

---

## 10. Real-Time Streaming Pipeline

### Architecture

```
[data_simulator.py]
  Reads bank.csv row by row
  Adds transaction_id, timestamp, random amount ($10–$15000)
  Writes 1 JSON file every 2 seconds
        │
        ▼  streaming/input/*.json
[spark_streaming.py — Spark Structured Streaming]
  Reads new files every 10 seconds (micro-batch)
  Applies feature engineering (event_time cast + engineered features)
  Applies fraud detection rules
  Applies subscription prediction
  Applies windowed aggregation (1-min sliding window / 30-sec slide)
        │
        ├──▼  streaming/output/*.json            (per-record enriched stream)
        └──▼  streaming/output_windowed/*.json    (rolling window aggregates)
[Enriched records with:]
  - fraud_flag: CLEAN / SUSPICIOUS
  - fraud_score: 0–100
  - predicted_subscription: YES / NO
  - subscription_confidence: 0.25–0.90
[Windowed aggregates with:]
  - window_start / window_end
  - tx_count, avg_amount, avg_balance, predicted_subscribers (per fraud_flag)
```

### Fraud Detection Rules

| Rule | Condition | Score |
|------|-----------|-------|
| Large transaction | amount > $10,000 | +40 |
| Negative balance + large tx | balance < 0 AND amount > $500 | +30 |
| Excessive contacts | campaign > 15 | +20 |
| Suspiciously short call | duration < 5 sec | +10 |

Total score ≥ 10 → flagged as `SUSPICIOUS`

### Window Operations

A sliding-window aggregation (`build_windowed_aggregates()` in `spark_streaming.py`) computes rolling, near-real-time KPIs on top of the per-record stream:

- **Window:** 1 minute wide, sliding every 30 seconds (`F.window("event_time", "1 minute", "30 seconds")`)
- **Watermark:** 2 minutes (`withWatermark("event_time", "2 minutes")`) — bounds how long Spark waits for late-arriving records before closing a window and releasing its state, preventing unbounded memory growth in a long-running stream
- **Grouped by:** the window plus `fraud_flag`, so clean vs. suspicious activity get separate rolling counts
- **Metrics:** transaction count, average transaction amount, average balance, predicted-subscriber count
- **Output modes:** `update` for the console sink (windows revise as data arrives within the watermark), `append` for the JSON file sink (only finalized/closed windows are written)

This produces the kind of rolling fraud/volume dashboard metric an ops team would actually monitor, rather than only a flat per-record log.

### Running the Streaming Pipeline

```bash
# Terminal 1: start data simulator
python3 streaming/data_simulator.py

# Terminal 2: start Spark Streaming consumer (includes windowed aggregation)
spark-submit streaming/spark_streaming.py
```

---

## 11. Data Parallelism & Optimization

### Partitioning in Spark

```
Dataset: 4521 rows
  ↓ repartition(8)
Partition 0: ~565 rows  [CPU Core 1]
Partition 1: ~565 rows  [CPU Core 2]
...
Partition 7: ~565 rows  [CPU Core 8]
  ↓ parallel processing
Results merged
```

Each partition is processed by one executor core **simultaneously** — this is data parallelism.

### Key Optimizations Applied

| Technique | What it does | Where used |
|-----------|-------------|------------|
| `spark.sql.shuffle.partitions=8` | Right-sizes shuffle for small data | All scripts |
| Parquet + Snappy | Columnar storage, 5× compression | Data storage |
| ORC in Hive | Predicate pushdown, column pruning | Hive tables |
| `handleInvalid='keep'` in encoders | Avoids job failure on unseen categories | Feature pipeline |
| `parallelism=2` in CrossValidator | Runs 2 CV folds simultaneously | Model tuning |
| `persist()` before multi-pass ops | Avoids recomputation of same RDD | Training loop |

### Spark Execution Optimization

```python
# Cache DataFrame that is used multiple times
train_df.persist()

# Broadcast small lookup tables (< 10 MB) to avoid shuffle
from pyspark.sql.functions import broadcast
df.join(broadcast(small_df), "key")

# Coalesce before writing to avoid tiny files
df.coalesce(4).write.parquet("output/")
```

---

## 12. Project Folder Structure

```
financial_forecasting_distributed_ml/
│
├── data/
│   ├── bank.csv                      ← Raw dataset
│   ├── bank_clean.parquet            ← After EDA cleaning
│   ├── train.parquet                 ← 80% split
│   ├── test.parquet                  ← 20% split
│   ├── feature_pipeline_model/       ← Saved Spark ML Pipeline
│   └── models/
│       ├── logistic_regression/
│       ├── decision_tree/
│       └── random_forest/            ← Best model
│
├── hdfs/
│   └── hdfs_setup.sh                 ← HDFS directory creation + upload
│
├── hive/
│   └── bank_hive.sql                 ← DDL + analytics queries
│
├── spark/
│   ├── eda.py                        ← Exploratory Data Analysis
│   ├── feature_engineering.py        ← Feature pipeline
│   └── model_training.py             ← LR + DT + RF + evaluation
│
├── streaming/
│   ├── spark_streaming.py            ← Structured Streaming consumer
│   └── data_simulator.py             ← Real-time data producer
│
├── notebooks/
│   ├── 01_data_ingestion_hadoop_hive.ipynb
│   ├── 02_eda_feature_engineering_spark.ipynb
│   ├── 03_model_training_validation_spark_ml.ipynb
│   ├── 04_streaming_window_operations.ipynb
│   └── 05_data_parallelism_optimization.ipynb
│
├── utils/
│   └── spark_utils.py                ← Shared helper functions
│
├── docs/
│   ├── plots/
│   │   ├── 01_target_distribution.png
│   │   ├── 02_sub_rate_by_job.png
│   │   ├── 03_age_distribution.png
│   │   ├── 04_model_comparison.png
│   │   └── 05_duration_impact.png
│   ├── model_results.json
│   ├── Video_Presentation_Script.docx ← Section-by-section script for the 15-min video
│   └── Reflective_Summary.docx        ← Challenges faced + learning outcomes
│
├── requirements.txt
└── README.md
```

---

## 13. Setup & Execution Guide

### Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| Java JDK | 8 or 11 | Hadoop + Spark runtime |
| Apache Hadoop | 3.3.x | HDFS + YARN |
| Apache Hive | 3.1.x | SQL query layer |
| Apache Spark | 3.5.x | Processing + ML |
| Anaconda | Latest | Environment & package management |
| Python | **3.10** (via Anaconda env `fin_distml`) | PySpark scripts |
| VS Code | Latest (launched via Anaconda Navigator or CLI) | IDE for development |

### Anaconda Environment Setup

```bash
# Create the dedicated Anaconda environment with Python 3.10
conda create -n fin_distml python=3.10 -y

# Activate the environment
conda activate fin_distml

# Install project dependencies inside the environment
pip install -r requirements.txt
```

> **VS Code Integration:** Open VS Code from within the activated environment, or select the `fin_distml` interpreter in VS Code via **Ctrl+Shift+P → Python: Select Interpreter**.

### Step 1: Install Dependencies

```bash
# Ensure the fin_distml environment is active before running this
pip install -r requirements.txt
```

### Step 2: Start Hadoop

```bash
# Format NameNode (first time only)
hdfs namenode -format

# Start HDFS
start-dfs.sh

# Start YARN (resource manager)
start-yarn.sh

# Verify
hdfs dfsadmin -report
```

### Step 3: HDFS Setup

```bash
chmod +x hdfs/hdfs_setup.sh
bash hdfs/hdfs_setup.sh
```

### Step 4: Hive Setup

```bash
# Start Hive Metastore
hive --service metastore &

# Run analytics
hive -f hive/bank_hive.sql
```

### Step 5: Run Spark Pipeline (in order)

```bash
# EDA
spark-submit spark/eda.py

# Feature Engineering
spark-submit spark/feature_engineering.py

# Model Training
spark-submit spark/model_training.py
```

### Step 6: Real-Time Streaming

```bash
# Terminal 1
python3 streaming/data_simulator.py

# Terminal 2
spark-submit streaming/spark_streaming.py
```

### Step 7: Local run (no Hadoop needed)

All scripts fall back to `local[*]` master and read from `data/` folder.
```bash
python3 spark/eda.py               # Works without Hadoop
python3 spark/feature_engineering.py
python3 spark/model_training.py
```

---

## 14. Interview & Viva Questions

### Hadoop / HDFS Questions

**Q1. What is the difference between HDFS and a regular file system?**  
HDFS stores files as large blocks (default 128 MB) distributed across multiple machines, with each block replicated 3 times for fault tolerance. Regular file systems store files on a single machine with no built-in replication. HDFS is designed for write-once, read-many sequential access patterns typical in batch analytics.

**Q2. What is the role of NameNode vs DataNode?**  
The NameNode is the master — it stores the filesystem metadata (directory tree, file-to-block mapping, block locations) entirely in RAM. DataNodes are workers that store the actual data blocks on disk and send heartbeats to the NameNode every 3 seconds. If a DataNode fails, the NameNode re-replicates its blocks from other replicas.

**Q3. What happens if the NameNode fails?**  
In older Hadoop (< 2.x), this was a single point of failure. Modern Hadoop uses High Availability (HA) with an Active and Standby NameNode. The Standby uses a shared edit log (on NFS or QJM) and ZooKeeper-based failover to take over in seconds.

**Q4. Why do we use Parquet/ORC instead of CSV in production?**  
CSV stores every value as text (no type awareness) and reads entire rows. Parquet and ORC are columnar formats — they store each column separately, apply column-level compression, and support predicate pushdown (skip irrelevant row groups). A typical analytics query that reads 3 of 17 columns processes 80% less data with Parquet.

---

### Apache Spark Questions

**Q5. What is the difference between RDD, DataFrame, and Dataset?**  
- **RDD** (Resilient Distributed Dataset): Low-level, type-safe, no query optimization, Java/Python objects. Use when you need custom partitioning or binary data.
- **DataFrame**: Distributed table with named columns and schema. Spark's Catalyst optimizer can rewrite and optimize SQL queries. Recommended for analytics.
- **Dataset**: Type-safe DataFrame (Scala/Java only). Python doesn't have Datasets — PySpark DataFrames behave like Datasets.

**Q6. Explain the difference between a transformation and an action.**  
Transformations (e.g., `filter()`, `select()`, `groupBy()`) are lazy — they define a computation plan but do not execute. Actions (e.g., `count()`, `show()`, `write()`) trigger execution. This laziness allows Spark to optimize the full DAG before running anything.

**Q7. What is the DAG in Spark?**  
DAG (Directed Acyclic Graph) is Spark's execution plan. Each action generates a DAG of stages. Each stage contains a sequence of transformations that can run without data shuffling. Shuffles (wide transformations) create stage boundaries. The DAG Scheduler converts this into physical tasks sent to executors.

**Q8. What is shuffle in Spark and why is it expensive?**  
Shuffle is the redistribution of data across partitions when operations like `groupBy`, `join`, or `orderBy` require all rows with the same key to be on the same executor. It involves serialization, disk writes, network transfer, and deserialization. Minimizing shuffles (via broadcast joins, partition pruning) is the primary Spark performance optimization.

**Q9. Explain `persist()` vs `cache()` in Spark.**  
`cache()` is shorthand for `persist(StorageLevel.MEMORY_AND_DISK)`. `persist()` allows you to choose the storage level: `MEMORY_ONLY` (fastest, fails if not enough RAM), `MEMORY_AND_DISK` (spills to disk), `DISK_ONLY` (slowest but handles any size). Use `cache()` for DataFrames used multiple times in model training.

**Q10. What are broadcast variables in Spark?**  
Broadcast variables send a read-only copy of a small dataset to every executor, avoiding repeated shipping with each task. Ideal for lookup tables < 10 MB. Example: broadcasting a country-code lookup table for a join with a billion-row transaction table reduces data transfer by orders of magnitude.

---

### Spark ML / Machine Learning Questions

**Q11. Why use Spark ML instead of scikit-learn?**  
scikit-learn runs on a single machine and loads data into RAM. Spark ML operates on distributed DataFrames across a cluster — it can train models on datasets that don't fit on one machine. For 4,521 rows, scikit-learn is faster; for 400 million rows, Spark ML is the only option.

**Q12. What is a Pipeline in Spark ML?**  
A Pipeline chains multiple Transformers and Estimators into a single workflow. A Transformer (e.g., `StringIndexer`, `StandardScaler`) applies a transformation to data. An Estimator (e.g., `LogisticRegression`) learns parameters from training data. Pipelines prevent data leakage (fitting scalers on test data) and make deployment clean.

**Q13. Why is ROC-AUC a better metric than accuracy for this dataset?**  
The dataset has an 88.5% class imbalance. A model that predicts "No" for every customer achieves 88.5% accuracy while being useless. ROC-AUC measures the model's ability to rank subscribers above non-subscribers across all thresholds — it is 0.5 for random guessing and 1.0 for perfect classification, regardless of class balance.

**Q14. Why did the Decision Tree have poor AUC (0.34) despite good accuracy?**  
Decision Trees can overfit to the training data when grown deep (maxDepth=8). They produce very confident predictions near 0 or 1, which makes the probability scores poorly calibrated for ranking. A single tree's probability estimates are based on the fraction of training samples in a leaf — this can be very noisy. Random Forest averages 100 trees, smoothing out these extremes and producing well-calibrated probabilities.

**Q15. Explain hyperparameter tuning with CrossValidator.**  
CrossValidator splits training data into k folds (we used k=3). For each hyperparameter combination, it trains on k-1 folds and validates on the remaining fold — k times. The combination with the best average validation metric (AUC) is selected. This prevents selection bias from a single train/validation split.

---

### Hive Questions

**Q16. What is the difference between EXTERNAL and MANAGED tables in Hive?**  
For a MANAGED table, Hive owns the data — dropping the table deletes the HDFS data. For an EXTERNAL table, Hive only stores metadata; dropping the table leaves HDFS data intact. Always use EXTERNAL for production data that other tools also read.

**Q17. What is ORC format and why is it used in Hive?**  
ORC (Optimized Row Columnar) is Hive's native columnar storage format. It supports predicate pushdown (skip row groups that don't match the WHERE clause without reading them), column-level compression (ZLIB, Snappy), bloom filters for fast lookups, and stores type statistics for each row group. Queries on ORC tables can be 10–100× faster than on text/CSV tables.

---

### Streaming Questions

**Q18. What is the difference between Spark Streaming (DStreams) and Structured Streaming?**  
DStreams (legacy) operate on RDDs in time-based micro-batches. Structured Streaming treats a live data stream as an unbounded DataFrame — the same DataFrame API works for both batch and streaming. Structured Streaming provides exactly-once semantics, event-time processing, and watermarking for late data handling.

**Q19. What is a checkpoint in Spark Streaming?**  
A checkpoint saves the streaming query's state (offset information, aggregation state) to a durable location (HDFS or local disk). If the streaming job crashes, it resumes from the last checkpoint rather than reprocessing from the beginning. Without checkpointing, stateful operations (like windowed aggregations) cannot recover from failures.

**Q20. What is watermarking in Structured Streaming?**  
Watermarking handles late-arriving records in event-time aggregations. You define a threshold (e.g., `withWatermark("timestamp", "10 minutes")`) — Spark keeps state for events up to 10 minutes late, then drops anything older. Without watermarking, Spark would keep state forever, causing memory exhaustion.

---

## 15. Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|---------|
| `WARN: Unable to load native-hadoop library` | Native library path not set | Set `HADOOP_HOME` and add to `LD_LIBRARY_PATH`. Add to `.bashrc`: `export HADOOP_HOME=/usr/local/hadoop` |
| `java.io.IOException: Connection refused (NameNode)` | NameNode not running | Run `start-dfs.sh`. Check with `jps` — should show `NameNode`, `DataNode` |
| `AnalysisException: 'default' is a reserved keyword` | Column named 'default' | Rename: `df.withColumnRenamed("default", "credit_default")` |
| `OutOfMemoryError: Java heap space` | Spark driver OOM | Increase `spark.driver.memory=4g` in SparkSession config |
| `ERROR: could not find or load main class` | SPARK_HOME not set | `export SPARK_HOME=/usr/local/spark; export PATH=$PATH:$SPARK_HOME/bin` |
| `ClassNotFoundException: org.apache.hive.jdbc.HiveDriver` | Hive JDBC jar missing | Add `$HIVE_HOME/lib/hive-jdbc-*.jar` to classpath |
| `WARN: Truncated the string representation` | DataFrame display width | Use `.show(truncate=False)` |
| `ERROR: Could not find a 'python3' executable` | PySpark can't find Python | `export PYSPARK_PYTHON=python3` |
| `FileNotFoundException on HDFS path` | Path doesn't exist | Run `hdfs dfs -ls /your/path` to verify; check for typos |
| `AnalysisException: Path does not exist` | Parquet output not created | Run `feature_engineering.py` before `model_training.py` |
| Streaming query stops immediately | No checkpoint directory | Create dir: `mkdir -p streaming/checkpoint` |
| `WARN: Unable to find encoder for type stored in a Dataset` | Complex type in DataFrame | Use `.select("label", "features")` before writing |
