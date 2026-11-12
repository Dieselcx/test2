
import pandas as pd
from pathlib import Path

# =========================
# Configuration
# =========================
INPUT_FILE = Path("/Users/shane/Desktop/test/resolution_sbert_kmeans/sbert_hdbscan_templates.csv")
OUTPUT_FILE = Path("/Users/shane/Desktop/test/resolution_sbert_kmeans/hdbscan_cluster_report.txt")

TOP_N_CLUSTER = 5
TOP_N_NOISE = 20


# =========================
# Load data
# =========================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

required_columns = {
    "cluster_id",
    "frequency",
    "original_text"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["cluster_id"] = pd.to_numeric(
    df["cluster_id"], errors="coerce"
)

df["frequency"] = pd.to_numeric(
    df["frequency"], errors="coerce"
).fillna(0)

df = df.dropna(subset=["cluster_id"])
df["cluster_id"] = df["cluster_id"].astype(int)

# Ignore invalid frequencies
df = df[df["frequency"] >= 0].copy()

# Total records represented by all templates
total_records = df["frequency"].sum()

# Separate Noise and normal clusters
noise_df = df[df["cluster_id"] == -1].copy()
cluster_df = df[df["cluster_id"] != -1].copy()

# =========================
# Generate report
# =========================

lines = []

def add(text=""):
    lines.append(str(text))


add("=" * 90)
add("HDBSCAN CLUSTER ANALYSIS REPORT")
add("=" * 90)

add(f"Total templates: {len(df):,}")
add(f"Total records: {total_records:,.0f}")
add(f"Number of clusters: {cluster_df['cluster_id'].nunique():,}")
add(f"Noise templates: {len(noise_df):,}")
add(
    f"Noise records: {noise_df['frequency'].sum():,.0f} "
    f"({noise_df['frequency'].sum() / total_records * 100:.2f}%)"
    if total_records > 0 else "Noise records: 0"
)

# =========================
# 1. Cluster overview
# =========================

add("\n\n" + "=" * 90)
add("1. CLUSTER OVERVIEW")
add("=" * 90)

overview = []

for cluster_id, group in cluster_df.groupby("cluster_id"):
    record_count = group["frequency"].sum()

    overview.append({
        "cluster_id": cluster_id,
        "record_count": record_count,
        "template_count": len(group),
        "percentage": (
            record_count / total_records * 100
            if total_records > 0 else 0
        )
    })

overview_df = pd.DataFrame(overview)

overview_df = overview_df.sort_values(
    "record_count", ascending=False
)

for _, row in overview_df.iterrows():
    add(
        f"Cluster {int(row['cluster_id']):>3} | "
        f"Records: {int(row['record_count']):>10,} | "
        f"Templates: {int(row['template_count']):>5,} | "
        f"Share: {row['percentage']:>6.2f}%"
    )

# =========================
# 2. Top templates per cluster
# =========================

add("\n\n" + "=" * 90)
add(f"2. TOP {TOP_N_CLUSTER} TEMPLATES PER CLUSTER")
add("=" * 90)

for cluster_id, group in cluster_df.groupby("cluster_id"):
    group = group.sort_values(
        "frequency", ascending=False
    )

    record_count = group["frequency"].sum()

    add("\n" + "-" * 90)
    add(
        f"Cluster {cluster_id} | "
        f"Records: {record_count:,.0f} | "
        f"Templates: {len(group):,}"
    )
    add("-" * 90)

    for i, (_, row) in enumerate(
        group.head(TOP_N_CLUSTER).iterrows(), 1
    ):
        add(
            f"\n{i}. [{int(row['frequency']):,} records]"
        )
        add(f"   {row['original_text']}")

# =========================
# 3. Top 20 Noise templates
# =========================

add("\n\n" + "=" * 90)
add(f"3. TOP {TOP_N_NOISE} NOISE TEMPLATES")
add("=" * 90)

noise_df = noise_df.sort_values(
    "frequency", ascending=False
)

for i, (_, row) in enumerate(
    noise_df.head(TOP_N_NOISE).iterrows(), 1
):
    add(f"\n{i}. [{int(row['frequency']):,} records]")
    add(f"   {row['original_text']}")

# =========================
# 4. Save report
# =========================

OUTPUT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("=" * 60)
print("HDBSCAN report generated successfully!")
print(f"Output: {OUTPUT_FILE.resolve()}")
print(f"Total records: {total_records:,.0f}")
print(f"Clusters: {cluster_df['cluster_id'].nunique()}")
print(f"Noise records: {noise_df['frequency'].sum():,.0f}")
print("=" * 60)
