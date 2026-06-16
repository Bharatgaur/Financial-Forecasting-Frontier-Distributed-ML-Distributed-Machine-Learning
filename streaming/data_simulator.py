# =============================================================================
# streaming/data_simulator.py  —  Simulates Real-Time Banking Transactions
# =============================================================================
# This script reads bank.csv and drops one JSON record per second into
# streaming/input/, simulating a live data feed (Kafka alternative for local use)
# Run:  python3 streaming/data_simulator.py
# =============================================================================

import csv, json, time, random, uuid, os
from datetime import datetime

INPUT_DIR = "streaming/input"
os.makedirs(INPUT_DIR, exist_ok=True)

# Load source data
with open("data/bank.csv", newline="") as f:
    reader = csv.DictReader(f)
    records = list(reader)

print(f"✓ Loaded {len(records)} source records")
print(f"  Dropping 1 record every 2 seconds into {INPUT_DIR}/")
print("  Press Ctrl+C to stop\n")

i = 0
while True:
    # Cycle through records (infinite loop)
    row = records[i % len(records)]

    # Build enriched transaction record
    tx = {
        "transaction_id":  str(uuid.uuid4())[:8].upper(),
        "timestamp":       datetime.utcnow().isoformat(),
        "age":             int(row["age"]),
        "job":             row["job"],
        "marital":         row["marital"],
        "education":       row["education"],
        "credit_default":  row["default"],
        "balance":         int(row["balance"]),
        "housing":         row["housing"],
        "loan":            row["loan"],
        "contact":         row["contact"],
        "day":             int(row["day"]),
        "month":           row["month"],
        "duration":        int(row["duration"]),
        "campaign":        int(row["campaign"]),
        "pdays":           int(row["pdays"]),
        "previous":        int(row["previous"]),
        "poutcome":        row["poutcome"],
        # Simulated transaction amount (not in original dataset)
        "amount":          round(random.uniform(10, 15000), 2),
    }

    # Write as a single-record JSON file
    filename = f"{INPUT_DIR}/tx_{tx['transaction_id']}_{int(time.time())}.json"
    with open(filename, "w") as out:
        json.dump(tx, out)

    print(f"  [{tx['timestamp'][:19]}]  TX {tx['transaction_id']}"
          f"  age={tx['age']}  balance={tx['balance']}  amount=${tx['amount']:.0f}")

    i += 1
    time.sleep(2)
