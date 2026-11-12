
import pandas as pd
from pathlib import Path

# ==========================================
# Configuration
# ==========================================

INPUT_FILE = Path("/Users/shane/Desktop/test/resolution_sbert_kmeans/sbert_hdbscan_templates.csv")
OUTPUT_FILE = Path("/Users/shane/Desktop/test/resolution_sbert_kmeans/resolution_manual_labeling.csv")

EXPECTED_TEMPLATES = 501


# ==========================================
# 1. Load HDBSCAN template results
# ==========================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}.\n"
        "Please put this script in the same directory "
        "as sbert_hdbscan_templates.csv."
    )

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

required_columns = [
    "normalized_text",
    "frequency",
    "original_text",
    "cluster_id"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in input CSV: {missing}"
    )


# ==========================================
# 2. Clean and validate
# ==========================================

df["frequency"] = pd.to_numeric(
    df["frequency"], errors="coerce"
)

df["cluster_id"] = pd.to_numeric(
    df["cluster_id"], errors="coerce"
)

df["normalized_text"] = (
    df["normalized_text"].fillna("").astype(str)
)

df["original_text"] = (
    df["original_text"].fillna("").astype(str)
)

# Remove invalid empty templates
df = df[
    (df["normalized_text"].str.strip() != "") &
    (df["frequency"].notna()) &
    (df["frequency"] > 0)
].copy()

# Each normalized template should appear only once
duplicates = df[
    df["normalized_text"].duplicated(keep=False)
]

if not duplicates.empty:
    raise ValueError(
        f"Found {len(duplicates)} duplicated template rows. "
        "Please check the input CSV before continuing."
    )

# ==========================================
# 3. Build manual labeling dataset
# ==========================================

result = pd.DataFrame()

result["template_id"] = [
    f"T{i:04d}" for i in range(1, len(df) + 1)
]

result["normalized_text"] = df["normalized_text"]
result["original_text"] = df["original_text"]
result["frequency"] = df["frequency"].astype(int)

result["hdbscan_cluster"] = (
    df["cluster_id"].apply(
        lambda x: int(x) if pd.notna(x) else ""
    )
)

# These fields will be filled during manual labeling
result["suggested_category"] = ""
result["final_category"] = ""
result["notes"] = ""

# Sort by frequency, most common templates first
result = result.sort_values(
    "frequency",
    ascending=False
).reset_index(drop=True)

# Reassign IDs after sorting
result["template_id"] = [
    f"T{i:04d}" for i in range(1, len(result) + 1)
]

# ==========================================
# 4. Export
# ==========================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

total_records = result["frequency"].sum()
noise_records = result.loc[
    result["hdbscan_cluster"] == -1,
    "frequency"
].sum()

print("=" * 60)
print("MANUAL LABELING DATASET GENERATED")
print("=" * 60)

print(f"Templates: {len(result):,}")
print(f"Total records: {total_records:,}")
print(f"Noise records: {noise_records:,}")
print(f"Output: {OUTPUT_FILE.resolve()}")

if len(result) != EXPECTED_TEMPLATES:
    print(
        f"\nWARNING: Expected {EXPECTED_TEMPLATES} templates, "
        f"but found {len(result)}."
    )
    print("Please verify that the input CSV is the latest version.")

print("\nTop 10 templates:")
print(
    result[
        ["template_id", "frequency", "hdbscan_cluster", "original_text"]
    ].head(10).to_string(index=False)
)

print("\nDone.")
