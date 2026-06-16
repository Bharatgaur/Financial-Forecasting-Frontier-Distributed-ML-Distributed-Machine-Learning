-- =============================================================================
-- HIVE SCHEMA & ANALYTICS QUERIES FOR BANK ML PROJECT
-- =============================================================================
-- How to run:
--   hive -f hive/bank_hive.sql
-- Or interactively:
--   hive
--   hive> source hive/bank_hive.sql;
-- =============================================================================


-- =============================================================================
-- SECTION 1: DATABASE & EXTERNAL TABLE CREATION
-- =============================================================================
-- We use an EXTERNAL table so Hive does NOT delete data when table is dropped.
-- The data lives in HDFS at /user/bankproject/raw/bank/
-- =============================================================================

CREATE DATABASE IF NOT EXISTS bankdb
  COMMENT 'Banking ML Project Database'
  LOCATION '/user/bankproject/hive/warehouse';

USE bankdb;

-- Drop if exists (for re-runs)
DROP TABLE IF EXISTS bank_raw;

-- External table pointing to HDFS CSV
CREATE EXTERNAL TABLE bank_raw (
  age        INT        COMMENT 'Customer age',
  job        STRING     COMMENT 'Job type: admin, technician, services, etc.',
  marital    STRING     COMMENT 'Marital status: married, single, divorced',
  education  STRING     COMMENT 'Education: primary, secondary, tertiary',
  def        STRING     COMMENT 'Credit default: yes/no (renamed from default)',
  balance    INT        COMMENT 'Average yearly account balance in euros',
  housing    STRING     COMMENT 'Has housing loan: yes/no',
  loan       STRING     COMMENT 'Has personal loan: yes/no',
  contact    STRING     COMMENT 'Contact type: cellular, telephone, unknown',
  day        INT        COMMENT 'Last contact day of month',
  month      STRING     COMMENT 'Last contact month',
  duration   INT        COMMENT 'Last contact duration in seconds',
  campaign   INT        COMMENT 'Number of contacts in this campaign',
  pdays      INT        COMMENT 'Days since last contact (-1 = not contacted)',
  previous   INT        COMMENT 'Contacts before this campaign',
  poutcome   STRING     COMMENT 'Previous campaign outcome',
  y          STRING     COMMENT 'Subscribed to term deposit: yes/no (TARGET)'
)
ROW FORMAT DELIMITED
  FIELDS TERMINATED BY ','
  LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/user/bankproject/raw/bank/'
TBLPROPERTIES ('skip.header.line.count'='1');


-- =============================================================================
-- SECTION 2: OPTIMIZED ORC TABLE (for faster queries)
-- =============================================================================
-- ORC (Optimized Row Columnar) compresses data and enables predicate pushdown.
-- This is the table Spark/Hive will use for analytics.
-- =============================================================================

DROP TABLE IF EXISTS bank_orc;

CREATE TABLE bank_orc
STORED AS ORC
TBLPROPERTIES ("orc.compress"="SNAPPY")
AS SELECT * FROM bank_raw;

-- Verify row count
SELECT COUNT(*) AS total_records FROM bank_orc;
-- Expected: 4521


-- =============================================================================
-- SECTION 3: CUSTOMER SEGMENTATION QUERIES
-- =============================================================================

-- Q1: Subscription rate by job type
SELECT
  job,
  COUNT(*)                                                       AS total_customers,
  SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END)                   AS subscribed,
  ROUND(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS subscription_rate_pct
FROM bank_orc
GROUP BY job
ORDER BY subscription_rate_pct DESC;


-- Q2: Subscription rate by education level
SELECT
  education,
  COUNT(*)                                                       AS total,
  SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END)                   AS subscribed,
  ROUND(AVG(balance), 2)                                        AS avg_balance,
  ROUND(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS sub_rate_pct
FROM bank_orc
GROUP BY education
ORDER BY sub_rate_pct DESC;


-- Q3: Age-group segmentation
SELECT
  CASE
    WHEN age < 25 THEN 'Youth (< 25)'
    WHEN age BETWEEN 25 AND 34 THEN 'Young Adult (25-34)'
    WHEN age BETWEEN 35 AND 44 THEN 'Mid Career (35-44)'
    WHEN age BETWEEN 45 AND 59 THEN 'Senior (45-59)'
    ELSE 'Retired (60+)'
  END AS age_group,
  COUNT(*)                                                       AS total,
  ROUND(AVG(balance), 0)                                        AS avg_balance,
  ROUND(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS sub_rate_pct
FROM bank_orc
GROUP BY
  CASE
    WHEN age < 25 THEN 'Youth (< 25)'
    WHEN age BETWEEN 25 AND 34 THEN 'Young Adult (25-34)'
    WHEN age BETWEEN 35 AND 44 THEN 'Mid Career (35-44)'
    WHEN age BETWEEN 45 AND 59 THEN 'Senior (45-59)'
    ELSE 'Retired (60+)'
  END
ORDER BY sub_rate_pct DESC;


-- =============================================================================
-- SECTION 4: LOAN ANALYSIS QUERIES
-- =============================================================================

-- Q4: Default risk by balance bucket
SELECT
  CASE
    WHEN balance < 0   THEN 'Negative'
    WHEN balance < 500 THEN 'Low (0-499)'
    WHEN balance < 2000 THEN 'Medium (500-1999)'
    ELSE 'High (2000+)'
  END AS balance_bucket,
  COUNT(*)                                                       AS total,
  SUM(CASE WHEN def = 'yes' THEN 1 ELSE 0 END)                 AS defaulters,
  SUM(CASE WHEN housing = 'yes' THEN 1 ELSE 0 END)             AS housing_loan_holders,
  SUM(CASE WHEN loan = 'yes' THEN 1 ELSE 0 END)                AS personal_loan_holders,
  ROUND(SUM(CASE WHEN def = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS default_rate_pct
FROM bank_orc
GROUP BY
  CASE
    WHEN balance < 0   THEN 'Negative'
    WHEN balance < 500 THEN 'Low (0-499)'
    WHEN balance < 2000 THEN 'Medium (500-1999)'
    ELSE 'High (2000+)'
  END
ORDER BY default_rate_pct DESC;


-- Q5: Loan holders vs subscription
SELECT
  housing,
  loan,
  COUNT(*)                                                       AS total,
  SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END)                   AS subscribed,
  ROUND(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS sub_rate_pct
FROM bank_orc
GROUP BY housing, loan
ORDER BY sub_rate_pct DESC;


-- =============================================================================
-- SECTION 5: CAMPAIGN PERFORMANCE QUERIES
-- =============================================================================

-- Q6: Campaign success rate by month
SELECT
  month,
  COUNT(*)                                                       AS contacts,
  SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END)                   AS successes,
  ROUND(AVG(duration), 0)                                       AS avg_call_duration_sec,
  ROUND(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS success_rate_pct
FROM bank_orc
GROUP BY month
ORDER BY success_rate_pct DESC;


-- Q7: Impact of call duration on subscription
SELECT
  CASE
    WHEN duration < 60   THEN '< 1 min'
    WHEN duration < 180  THEN '1-3 min'
    WHEN duration < 300  THEN '3-5 min'
    WHEN duration < 600  THEN '5-10 min'
    ELSE '> 10 min'
  END AS duration_bucket,
  COUNT(*)                                                       AS total,
  SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END)                   AS subscribed,
  ROUND(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS sub_rate_pct
FROM bank_orc
GROUP BY
  CASE
    WHEN duration < 60   THEN '< 1 min'
    WHEN duration < 180  THEN '1-3 min'
    WHEN duration < 300  THEN '3-5 min'
    WHEN duration < 600  THEN '5-10 min'
    ELSE '> 10 min'
  END
ORDER BY sub_rate_pct DESC;


-- Q8: Previous campaign outcome vs current success
SELECT
  poutcome,
  COUNT(*)                                                       AS total,
  SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END)                   AS current_success,
  ROUND(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS success_rate_pct
FROM bank_orc
GROUP BY poutcome
ORDER BY success_rate_pct DESC;


-- Q9: Optimal number of contacts
SELECT
  campaign                                                       AS num_contacts,
  COUNT(*)                                                       AS total,
  ROUND(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*), 2)                                           AS sub_rate_pct
FROM bank_orc
GROUP BY campaign
HAVING COUNT(*) > 20
ORDER BY sub_rate_pct DESC
LIMIT 10;
