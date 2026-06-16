#!/bin/bash
# =============================================================================
# HDFS Setup Script for Bank ML Project
# =============================================================================
# This script creates the full HDFS folder structure and uploads the dataset.
# Run AFTER Hadoop is started: start-dfs.sh && start-yarn.sh
# =============================================================================

echo "======================================================"
echo "  BANK PROJECT - HDFS SETUP"
echo "======================================================"

# ------------------------------------------------------------------------------
# STEP 1: Create HDFS directory structure
# ------------------------------------------------------------------------------
# hdfs dfs -mkdir -p  →  like "mkdir -p" but on HDFS
# /user/bankproject   →  root namespace for this project
# ------------------------------------------------------------------------------

echo "[1/5] Creating HDFS directory structure..."

hdfs dfs -mkdir -p /user/bankproject/raw/bank
hdfs dfs -mkdir -p /user/bankproject/processed/features
hdfs dfs -mkdir -p /user/bankproject/processed/encoded
hdfs dfs -mkdir -p /user/bankproject/models/logistic_regression
hdfs dfs -mkdir -p /user/bankproject/models/decision_tree
hdfs dfs -mkdir -p /user/bankproject/models/random_forest
hdfs dfs -mkdir -p /user/bankproject/hive/warehouse
hdfs dfs -mkdir -p /user/bankproject/streaming/input
hdfs dfs -mkdir -p /user/bankproject/streaming/output
hdfs dfs -mkdir -p /user/bankproject/reports

echo "[1/5] ✓ Directory structure created."

# ------------------------------------------------------------------------------
# STEP 2: Upload dataset to HDFS
# ------------------------------------------------------------------------------
# hdfs dfs -put  →  upload a local file to HDFS
# -f             →  overwrite if exists
# ------------------------------------------------------------------------------

echo "[2/5] Uploading bank.csv to HDFS..."

hdfs dfs -put -f data/bank.csv /user/bankproject/raw/bank/bank.csv

echo "[2/5] ✓ Dataset uploaded."

# ------------------------------------------------------------------------------
# STEP 3: Verify the upload
# ------------------------------------------------------------------------------

echo "[3/5] Verifying upload..."

hdfs dfs -ls /user/bankproject/raw/bank/
hdfs dfs -du -h /user/bankproject/raw/bank/

echo "[3/5] ✓ Verified."

# ------------------------------------------------------------------------------
# STEP 4: Set correct permissions
# ------------------------------------------------------------------------------

echo "[4/5] Setting permissions..."

hdfs dfs -chmod -R 755 /user/bankproject

echo "[4/5] ✓ Permissions set."

# ------------------------------------------------------------------------------
# STEP 5: Print full HDFS tree
# ------------------------------------------------------------------------------

echo "[5/5] HDFS Directory Tree:"
echo ""
echo "  /user/bankproject/"
echo "  ├── raw/"
echo "  │   └── bank/"
echo "  │       └── bank.csv         ← Raw input dataset"
echo "  ├── processed/"
echo "  │   ├── features/            ← After EDA + feature engineering"
echo "  │   └── encoded/             ← After encoding for ML"
echo "  ├── models/"
echo "  │   ├── logistic_regression/ ← Saved Spark ML model"
echo "  │   ├── decision_tree/       ← Saved Spark ML model"
echo "  │   └── random_forest/       ← Best model"
echo "  ├── hive/"
echo "  │   └── warehouse/           ← Hive managed tables"
echo "  ├── streaming/"
echo "  │   ├── input/               ← Incoming real-time records"
echo "  │   └── output/              ← Streaming predictions"
echo "  └── reports/                 ← Summary statistics & outputs"

echo ""
echo "======================================================"
echo "  HDFS SETUP COMPLETE"
echo "======================================================"
