# =============================================================================
# spark/model_training.py  —  ML Model Training & Evaluation using Spark ML
# =============================================================================
# Run:  spark-submit spark/model_training.py
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import json, os

# =============================================================================
# 1. SPARK SESSION
# =============================================================================
spark = (
    SparkSession.builder
    .appName("BankModelTraining")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "3g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# =============================================================================
# 2. LOAD PROCESSED DATA  (output from feature_engineering.py)
# =============================================================================
train_df = spark.read.parquet("data/train.parquet")
test_df  = spark.read.parquet("data/test.parquet")

print(f"✓ Train: {train_df.count()} rows  |  Test: {test_df.count()} rows")

# =============================================================================
# 3. EVALUATORS
# =============================================================================
# We need two evaluators because PySpark splits binary metrics across classes:
# BinaryClassificationEvaluator  → ROC-AUC
# MulticlassClassificationEvaluator → Accuracy, Precision, Recall, F1

binary_eval = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

def multi_eval(metric):
    return MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName=metric
    )

def evaluate_model(predictions, model_name):
    """Compute all five metrics for a set of predictions."""
    auc       = binary_eval.evaluate(predictions)
    accuracy  = multi_eval("accuracy").evaluate(predictions)
    precision = multi_eval("weightedPrecision").evaluate(predictions)
    recall    = multi_eval("weightedRecall").evaluate(predictions)
    f1        = multi_eval("f1").evaluate(predictions)

    print(f"\n{'='*50}")
    print(f"  {model_name} RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"{'='*50}")

    return {
        "model":     model_name,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "auc":       round(auc,       4)
    }

results = []   # collect all model metrics for final comparison

# =============================================================================
# 4. MODEL 1: LOGISTIC REGRESSION
# =============================================================================
# Logistic Regression is the baseline model for binary classification.
# regParam controls L2 regularization (prevents overfitting).
# elasticNetParam=0 → pure L2 (Ridge); 1 → pure L1 (Lasso).
# maxIter → max gradient descent iterations.
# =============================================================================

print("\n[1/3] Training Logistic Regression…")

lr = LogisticRegression(
    featuresCol    = "features",
    labelCol       = "label",
    maxIter        = 100,
    regParam       = 0.01,
    elasticNetParam= 0.0,
    family         = "binomial"
)

lr_model = lr.fit(train_df)
lr_preds = lr_model.transform(test_df)
lr_metrics = evaluate_model(lr_preds, "Logistic Regression")
results.append(lr_metrics)

# Confusion matrix
print("\n  Confusion Matrix (LR):")
(
    lr_preds
    .groupBy("label", "prediction")
    .count()
    .orderBy("label", "prediction")
    .show()
)

# Save model
lr_model.write().overwrite().save("data/models/logistic_regression")
print("✓ LR model saved")

# =============================================================================
# 5. MODEL 2: DECISION TREE
# =============================================================================
# Decision Trees split data on the feature that maximizes information gain.
# maxDepth controls tree complexity (deeper = more complex, risk of overfit).
# impurity='gini' → Gini impurity criterion (alternative: 'entropy')
# =============================================================================

print("\n[2/3] Training Decision Tree…")

dt = DecisionTreeClassifier(
    featuresCol = "features",
    labelCol    = "label",
    maxDepth    = 8,
    impurity    = "gini",
    seed        = 42
)

dt_model = dt.fit(train_df)
dt_preds = dt_model.transform(test_df)
dt_metrics = evaluate_model(dt_preds, "Decision Tree")
results.append(dt_metrics)

# Feature importances (top 10)
print("\n  Top 10 Feature Importances (Decision Tree):")
importances = dt_model.featureImportances
top10 = sorted(
    enumerate(importances.toArray()), key=lambda x: -x[1]
)[:10]
for rank, (idx, imp) in enumerate(top10, 1):
    print(f"    #{rank}  feature[{idx:3d}]  importance={imp:.4f}")

dt_model.write().overwrite().save("data/models/decision_tree")
print("✓ DT model saved")

# =============================================================================
# 6. MODEL 3: RANDOM FOREST  (with hyperparameter tuning)
# =============================================================================
# Random Forest = ensemble of Decision Trees with bagging + random feature subsets.
# numTrees → more trees = better but slower
# maxDepth → depth of each individual tree
# We tune these with 3-fold CrossValidator to find the best combination.
# =============================================================================

print("\n[3/3] Training Random Forest (with CrossValidator tuning)…")

rf = RandomForestClassifier(
    featuresCol = "features",
    labelCol    = "label",
    seed        = 42
)

# Hyperparameter grid
param_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees,  [50, 100])      # 2 values
    .addGrid(rf.maxDepth,  [5, 10])        # 2 values
    .build()
)
# → 4 combinations × 3 folds = 12 total fits

cv = CrossValidator(
    estimator          = rf,
    estimatorParamMaps = param_grid,
    evaluator          = binary_eval,      # optimize for ROC-AUC
    numFolds           = 3,
    seed               = 42,
    parallelism        = 2                 # run 2 folds in parallel
)

print("  Running 3-fold cross-validation (12 fits)… this may take ~2 minutes")
cv_model = cv.fit(train_df)
best_rf   = cv_model.bestModel

# Best hyperparameters
best_trees = best_rf.getNumTrees
best_depth = best_rf.getOrDefault("maxDepth")
print(f"  Best params → numTrees={best_trees}, maxDepth={best_depth}")

rf_preds   = best_rf.transform(test_df)
rf_metrics = evaluate_model(rf_preds, "Random Forest (Best)")
results.append(rf_metrics)

# Feature importances
print("\n  Top 10 Feature Importances (Random Forest):")
rf_importances = best_rf.featureImportances
rf_top10 = sorted(
    enumerate(rf_importances.toArray()), key=lambda x: -x[1]
)[:10]
for rank, (idx, imp) in enumerate(rf_top10, 1):
    print(f"    #{rank}  feature[{idx:3d}]  importance={imp:.4f}")

best_rf.write().overwrite().save("data/models/random_forest")
print("✓ RF model saved")

# =============================================================================
# 7. MODEL COMPARISON TABLE
# =============================================================================
print("\n" + "="*70)
print("  MODEL COMPARISON SUMMARY")
print("="*70)
print(f"  {'Model':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("-"*70)
for r in results:
    print(f"  {r['model']:<30} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f}")
print("="*70)

# Winner
best = max(results, key=lambda x: x["auc"])
print(f"\n  🏆 BEST MODEL: {best['model']} (AUC = {best['auc']:.4f})")

# Save results JSON for reporting
os.makedirs("docs", exist_ok=True)
with open("docs/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✓ Results saved → docs/model_results.json")

# =============================================================================
# 8. THRESHOLD ANALYSIS ON BEST MODEL
# =============================================================================
# Default Spark ML decision threshold = 0.5.
# In banking, we often prefer higher recall (catch more potential subscribers).
# Let's evaluate at threshold = 0.3 for the best model (RF).

print("\n--- Threshold Analysis (Random Forest @ threshold=0.3) ---")

from pyspark.ml.classification import RandomForestClassificationModel
rf_saved = RandomForestClassificationModel.load("data/models/random_forest")
rf_preds_all = rf_saved.transform(test_df)

# Extract probability of class 1 (subscribed=yes)
extract_prob = F.udf(lambda v: float(v[1]))
rf_thresh = (
    rf_preds_all
    .withColumn("prob_yes", extract_prob(F.col("probability")))
    .withColumn("pred_thresh03",
        F.when(F.col("prob_yes") >= 0.3, 1.0).otherwise(0.0))
)

tp = rf_thresh.filter((F.col("pred_thresh03")==1) & (F.col("label")==1)).count()
fp = rf_thresh.filter((F.col("pred_thresh03")==1) & (F.col("label")==0)).count()
tn = rf_thresh.filter((F.col("pred_thresh03")==0) & (F.col("label")==0)).count()
fn = rf_thresh.filter((F.col("pred_thresh03")==0) & (F.col("label")==1)).count()

prec_t = tp / (tp + fp) if (tp+fp) > 0 else 0
rec_t  = tp / (tp + fn) if (tp+fn) > 0 else 0
f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t+rec_t) > 0 else 0

print(f"  Threshold=0.3 → Precision={prec_t:.4f}  Recall={rec_t:.4f}  F1={f1_t:.4f}")
print(f"  Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")

spark.stop()
print("\n✓ MODEL TRAINING COMPLETE")
