#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NYC 311 通用数据预处理

功能：
1. 递归读取 JSONL / JSONL.GZ 文件
2. 分块读取，避免一次性加载全部数据
3. 统一字段和数据类型
4. 删除 unique_key 缺失、created_date 无效的记录
5. 使用 SQLite 对所有输入文件进行全局去重
6. 生成时间、地理位置和解决效率相关特征
7. 输出按 year 分区的 Parquet 数据集
8. 生成总体数据质量报告

注意：
- complaint_type、descriptor、agency 等字段缺失不会导致记录被删除。
- closed_date 缺失不会导致记录被删除。
- closed_date 早于 created_date 的记录会保留在通用数据集中。
- descriptor_2 不属于当前项目的研究范围。
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import shutil
import sqlite3
import sys
import tempfile
import time

from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 1. 配置
# ============================================================

SELECTED_COLUMNS = [
    "unique_key",

    # 投诉类型
    "complaint_type",
    "descriptor",
    "agency",
    "agency_name",
    "status",

    # 时间
    "created_date",
    "closed_date",
    "resolution_action_updated_date",
    "due_date",

    # 地理位置
    "borough",
    "incident_zip",
    "community_board",
    "council_district",
    "latitude",
    "longitude",

    # 其他信息
    "location_type",
    "resolution_description",
    "open_data_channel_type",
    "incident_address",
    "street_name",
    "address_type",
    "park_borough",
    "park_facility_name",
    "vehicle_type",
    "bridge_highway_name",
    "bridge_highway_direction",
    "road_ramp",
    "bridge_highway_segment",
]

DATETIME_COLUMNS = [
    "created_date",
    "closed_date",
    "resolution_action_updated_date",
    "due_date",
]

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
]

STRING_COLUMNS = [
    c for c in SELECTED_COLUMNS
    if c not in DATETIME_COLUMNS + NUMERIC_COLUMNS
]

VALID_BOROUGHS = {
    "BRONX",
    "BROOKLYN",
    "MANHATTAN",
    "QUEENS",
    "STATEN ISLAND",
}

LOG = logging.getLogger("nyc311")


# ============================================================
# 2. 工具函数
# ============================================================

def is_blank(series: pd.Series) -> pd.Series:
    """判断字段是否为空，包括 None、空字符串和纯空格。"""
    return (
        series.isna()
        | series.astype("string").str.strip().eq("").fillna(True)
    )


def parse_datetime(series: pd.Series) -> pd.Series:
    """解析时间字段，无法解析的值转为 NaT。"""
    return pd.to_datetime(
        series,
        errors="coerce",
        format="mixed"
    )


def normalize_borough(series: pd.Series) -> pd.Series:
    """标准化 Borough 名称，不猜测未知值。"""
    result = series.astype("string").str.strip().str.upper()

    invalid_values = [
        "",
        "UNSPECIFIED",
        "UNKNOWN",
        "N/A",
        "NA",
        "NULL",
    ]

    result = result.mask(result.isin(invalid_values))
    return result


def infer_year(file_path: Path) -> str:
    """
    尝试从文件名或父目录获取年份。
    例如：
        2025_01.jsonl.gz -> 2025
        /raw_data/2025/01.jsonl.gz -> 2025
    """
    candidates = [file_path.stem, file_path.parent.name]

    for value in candidates:
        match = re.search(r"(20\d{2})", value)
        if match:
            return match.group(1)

    return "unknown"


def open_jsonl_file(file_path: Path):
    """根据文件后缀打开普通 JSONL 或 GZIP JSONL。"""
    if file_path.name.endswith(".gz"):
        return gzip.open(file_path, mode="rt", encoding="utf-8")

    return open(file_path, mode="rt", encoding="utf-8")


# ============================================================
# 3. SQLite 全局去重
# ============================================================

def init_dedup_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))

    # 使用 DELETE 模式，避免 WAL/SHM 文件与云盘同步产生冲突。
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_keys (
            unique_key TEXT PRIMARY KEY
        )
    """)

    conn.commit()
    return conn


def remove_global_duplicates(
    df: pd.DataFrame,
    conn: sqlite3.Connection
) -> tuple[pd.DataFrame, int]:
    """
    对当前 chunk 执行全局去重。

    规则：
    - 同一 chunk 内重复：保留第一次出现的记录。
    - 不同文件或不同 chunk 重复：保留最先处理的有效记录。
    """

    before = len(df)

    df = df.drop_duplicates(
        subset=["unique_key"],
        keep="first"
    ).copy()

    local_duplicates = before - len(df)

    keys = df["unique_key"].astype(str).tolist()

    existing_keys = set()

    # 分批查询，避免 SQLite 参数数量限制。
    batch_size = 800

    for start in range(0, len(keys), batch_size):
        batch = keys[start:start + batch_size]

        placeholders = ",".join(["?"] * len(batch))

        sql = (
            "SELECT unique_key FROM seen_keys "
            f"WHERE unique_key IN ({placeholders})"
        )

        rows = conn.execute(sql, batch).fetchall()

        existing_keys.update(row[0] for row in rows)

    duplicate_mask = df["unique_key"].astype(str).isin(existing_keys)

    global_duplicates = int(duplicate_mask.sum())

    df = df.loc[~duplicate_mask].copy()

    new_keys = df["unique_key"].astype(str).tolist()

    if new_keys:
        conn.executemany(
            "INSERT OR IGNORE INTO seen_keys(unique_key) VALUES (?)",
            [(key,) for key in new_keys]
        )
        conn.commit()

    return df, local_duplicates + global_duplicates


# ============================================================
# 4. 特征工程
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    # ----------------------------
    # 时间特征
    # ----------------------------

    created = df["created_date"]

    df["year"] = created.dt.year.astype("int16")
    df["month"] = created.dt.month.astype("int8")

    df["year_month"] = (
        created.dt.strftime("%Y-%m").astype("string")
    )

    df["quarter"] = (
        "Q" + created.dt.quarter.astype("string")
    ).astype("string")

    df["day_of_week"] = (
        created.dt.day_name().astype("string")
    )

    df["hour"] = created.dt.hour.astype("int8")

    df["is_weekend"] = (
        created.dt.dayofweek >= 5
    ).astype(bool)

    # ----------------------------
    # 地理特征
    # ----------------------------

    df["borough_clean"] = normalize_borough(df["borough"])

    df["zipcode_clean"] = (
        df["incident_zip"]
        .astype("string")
        .str.strip()
    )

    df["is_zipcode_valid"] = (
        df["zipcode_clean"]
        .str.fullmatch(r"\d{5}", na=False)
    ).astype(bool)

    df["has_valid_borough"] = (
        df["borough_clean"].isin(VALID_BOROUGHS)
    ).astype(bool)

    df["is_location_valid"] = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ).astype(bool)

    df["has_valid_geography"] = (
        df["has_valid_borough"]
        | df["is_zipcode_valid"]
        | df["is_location_valid"]
    ).astype(bool)

    # ----------------------------
    # 解决效率特征
    # ----------------------------

    # 是否存在关闭时间记录。
    df["has_closed_date"] = (
        df["closed_date"].notna()
    ).astype(bool)

    # 这里的 is_closed 表示存在关闭时间记录，
    # 不直接根据 status 推断。
    df["is_closed"] = df["has_closed_date"]

    # 是否尚无关闭时间记录。
    df["is_pending"] = (
        ~df["has_closed_date"]
    ).astype(bool)

    # 关闭时间是否早于创建时间。
    df["closed_before_created"] = (
        df["closed_date"].notna()
        & (df["closed_date"] < df["created_date"])
    ).astype(bool)

    # 解决时间是否有效。
    # 通用数据集中，创建时间已经通过前面的基础筛选。
    df["is_resolution_valid"] = (
        ~df["closed_before_created"]
    ).astype(bool)

    # 只有关闭时间有效时才计算解决时长。
    duration = (
        df["closed_date"] - df["created_date"]
    ).dt.total_seconds() / 3600

    df["resolution_hours"] = (
        duration.where(df["is_resolution_valid"])
        .where(df["has_closed_date"])
        .astype("float64")
    )

    df["resolution_days"] = (
        df["resolution_hours"] / 24
    ).astype("float64")

    return df


# ============================================================
# 5. 单个 chunk 的处理
# ============================================================

def process_chunk(
    raw_df: pd.DataFrame,
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, dict]:

    raw_count = len(raw_df)

    df = raw_df.copy()

    # 确保所有文件具有相同字段。
    for col in SELECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[SELECTED_COLUMNS].copy()

    # 字符串字段统一处理。
    for col in STRING_COLUMNS:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # 数值字段。
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).astype("float64")

    # 时间字段。
    for col in DATETIME_COLUMNS:
        df[col] = parse_datetime(df[col])

    # 删除 unique_key 缺失记录。
    missing_key = is_blank(df["unique_key"])

    missing_key_count = int(missing_key.sum())

    df = df.loc[~missing_key].copy()

    # 删除 created_date 缺失或无效记录。
    invalid_created = df["created_date"].isna()

    invalid_created_count = int(invalid_created.sum())

    df = df.loc[~invalid_created].copy()

    # 全局去重。
    df, duplicate_count = remove_global_duplicates(
        df,
        conn
    )

    # 生成特征。
    df = add_features(df)

    # 统计异常。
    anomalies = {
        "closed_before_created": int(
            df["closed_before_created"].sum()
        ),
        "missing_closed_date": int(
            df["closed_date"].isna().sum()
        ),
        "missing_borough": int(
            is_blank(df["borough"]).sum()
        ),
        "missing_latitude": int(
            df["latitude"].isna().sum()
        ),
        "missing_longitude": int(
            df["longitude"].isna().sum()
        ),
    }

    stats = {
        "raw_records": raw_count,
        "retained_records": len(df),
        "missing_unique_key": missing_key_count,
        "invalid_created_date": invalid_created_count,
        "duplicate_unique_key": duplicate_count,
        "anomalies": anomalies,
    }

    return df, stats


# ============================================================
# 6. Parquet 写入
# ============================================================

def write_parquet_chunk(
    df: pd.DataFrame,
    output_path: Path,
    file_index: int,
    chunk_index: int,
) -> None:

    if df.empty:
        return

    table = pa.Table.from_pandas(
        df,
        preserve_index=False
    )

    # 每个 chunk 使用唯一文件名，避免覆盖之前写入的数据。
    basename_template = (
        f"part-{file_index:04d}-{chunk_index:06d}-{{i}}.parquet"
    )

    pq.write_to_dataset(
        table,
        root_path=str(output_path),
        partition_cols=["year"],
        compression="snappy",
        basename_template=basename_template,
        existing_data_behavior="overwrite_or_ignore",
    )


# ============================================================
# 7. 单个文件处理
# ============================================================

def process_file(
    file_path: Path,
    file_index: int,
    output_path: Path,
    conn: sqlite3.Connection,
    chunksize: int,
    overall_stats: dict,
) -> None:

    LOG.info("Processing: %s", file_path)

    file_stats = {
        "file": str(file_path),
        "year": infer_year(file_path),
        "raw_records": 0,
        "retained_records": 0,
        "missing_unique_key": 0,
        "invalid_created_date": 0,
        "duplicate_unique_key": 0,
        "closed_before_created": 0,
        "missing_closed_date": 0,
        "missing_borough": 0,
        "missing_latitude": 0,
        "missing_longitude": 0,
    }

    chunk_index = 0

    # pandas 逐 chunk 读取 JSONL。
    with open_jsonl_file(file_path) as file_obj:

        reader = pd.read_json(
            file_obj,
            lines=True,
            chunksize=chunksize
        )

        for raw_df in reader:

            chunk_index += 1

            cleaned_df, stats = process_chunk(
                raw_df,
                conn
            )

            write_parquet_chunk(
                cleaned_df,
                output_path,
                file_index,
                chunk_index
            )

            file_stats["raw_records"] += stats["raw_records"]
            file_stats["retained_records"] += stats["retained_records"]
            file_stats["missing_unique_key"] += stats["missing_unique_key"]
            file_stats["invalid_created_date"] += stats["invalid_created_date"]
            file_stats["duplicate_unique_key"] += stats["duplicate_unique_key"]

            for key, value in stats["anomalies"].items():
                file_stats[key] += value

            LOG.info(
                "File %s | Chunk %d | Raw: %s | Retained: %s",
                file_path.name,
                chunk_index,
                f"{stats['raw_records']:,}",
                f"{stats['retained_records']:,}",
            )

    # 累计总体统计。
    for key in file_stats:
        if key in overall_stats:
            overall_stats[key] += file_stats[key]

    overall_stats["files"].append(file_stats)

    LOG.info(
        "Finished %s | Raw: %s | Retained: %s",
        file_path.name,
        f"{file_stats['raw_records']:,}",
        f"{file_stats['retained_records']:,}",
    )


# ============================================================
# 8. 质量报告
# ============================================================

def generate_quality_report(
    output_dir: Path,
    stats: dict,
) -> None:

    report_dir = output_dir / "quality_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    raw = stats["raw_records"]
    retained = stats["retained_records"]

    report = {
        "dataset": "NYC 311 Service Requests",
        "raw_records": raw,
        "retained_records": retained,
        "dropped_records": raw - retained,
        "retention_rate": (
            round(retained / raw, 6) if raw else None
        ),
        "input_file_count": len(stats["files"]),
        "missing_unique_key": stats["missing_unique_key"],
        "invalid_created_date": stats["invalid_created_date"],
        "duplicate_unique_key": stats["duplicate_unique_key"],
        "anomalies": {
            "closed_before_created": stats["closed_before_created"],
            "missing_closed_date": stats["missing_closed_date"],
            "missing_borough": stats["missing_borough"],
            "missing_latitude": stats["missing_latitude"],
            "missing_longitude": stats["missing_longitude"],
        },
        "files": stats["files"],
        "rules": [
            "Missing unique_key records are removed.",
            "Missing or invalid created_date records are removed.",
            "Global deduplication is performed using unique_key.",
            "Missing complaint_type, descriptor, agency and status are preserved.",
            "Missing closed_date records are preserved.",
            "Records with closed_date earlier than created_date are preserved in the canonical dataset.",
            "Negative resolution durations are set to null.",
            "descriptor_2 is excluded from the project dataset.",
            "Coordinate validation checks global legal ranges, not NYC boundary membership.",
        ],
    }

    json_path = report_dir / "overall_report.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    retention = (
        retained / raw * 100
        if raw else 0
    )

    md = [
        "# NYC 311 数据质量报告",
        "",
        "## 1. 总体处理结果",
        "",
        f"- 输入文件数：{len(stats['files']):,}",
        f"- 原始记录数：{raw:,}",
        f"- 保留记录数：{retained:,}",
        f"- 删除记录数：{raw - retained:,}",
        f"- 数据保留率：{retention:.2f}%",
        f"- 缺失 unique_key：{stats['missing_unique_key']:,}",
        f"- created_date 无效：{stats['invalid_created_date']:,}",
        f"- 重复 unique_key：{stats['duplicate_unique_key']:,}",
        "",
        "## 2. 异常与缺失情况",
        "",
        f"- 关闭时间早于创建时间：{stats['closed_before_created']:,}",
        f"- 缺失关闭时间：{stats['missing_closed_date']:,}",
        f"- 缺失 Borough：{stats['missing_borough']:,}",
        f"- 缺失 Latitude：{stats['missing_latitude']:,}",
        f"- 缺失 Longitude：{stats['missing_longitude']:,}",
        "",
        "## 3. 数据处理规则",
        "",
        "- 缺失 unique_key 的记录会被删除。",
        "- 缺失或无法解析 created_date 的记录会被删除。",
        "- 按 unique_key 在所有输入文件范围内全局去重。",
        "- complaint_type、descriptor、agency 等字段缺失不会导致记录被删除。",
        "- closed_date 缺失不会导致记录被删除。",
        "- closed_date 早于 created_date 的记录保留在通用数据集中。",
        "- 对于关闭时间异常的记录，resolution_hours 和 resolution_days 为空。",
        "- descriptor_2 不属于当前项目研究范围。",
        "",
        "## 4. 输出目录",
        "",
        "```text",
        "processed_data/",
        "quality_reports/",
        "    overall_report.json",
        "    README_质量报告.md",
        "```",
    ]

    (report_dir / "README_质量报告.md").write_text(
        "\n".join(md),
        encoding="utf-8"
    )


# ============================================================
# 9. 主程序
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NYC 311 通用数据预处理"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="原始 JSONL / JSONL.GZ 数据目录"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="预处理结果输出目录"
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="每个 chunk 读取的记录数量"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="删除已有输出目录后重新处理"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    if args.chunksize <= 0:
        raise ValueError("--chunksize 必须大于 0")

    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("输出目录不能是输入目录或位于输入目录内部")

    files = sorted(
        list(input_dir.rglob("*.jsonl.gz"))
        + list(input_dir.rglob("*.jsonl"))
    )

    if not files:
        raise FileNotFoundError(
            f"没有在目录中找到 JSONL 文件：{input_dir}"
        )

    if output_dir.exists():

        if args.overwrite:
            shutil.rmtree(output_dir)

        elif any(output_dir.iterdir()):
            raise FileExistsError(
                "输出目录非空。如需重新处理，请更换目录或使用 --overwrite"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    parquet_dir = output_dir / "processed_data"
    parquet_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG.info("Found %d input files.", len(files))
    LOG.info("Output directory: %s", output_dir)
    
    overall_stats = {
    "raw_records": 0,
    "retained_records": 0,
    "missing_unique_key": 0,
    "invalid_created_date": 0,
    "duplicate_unique_key": 0,
    "closed_before_created": 0,
    "missing_closed_date": 0,
    "missing_borough": 0,
    "missing_latitude": 0,
    "missing_longitude": 0,
    "files": [],
}

    start_time = time.perf_counter()

    # SQLite 临时数据库放在系统临时目录，
    # 避免 OneDrive 等同步目录影响 SQLite 写入。
    with tempfile.TemporaryDirectory(
        prefix="nyc311_dedup_"
    ) as temp_dir:

        db_path = Path(temp_dir) / "dedup.sqlite"

        conn = init_dedup_db(db_path)

        try:

            for index, file_path in enumerate(files, start=1):

                LOG.info(
                    "========== File %d/%d ==========",
                    index,
                    len(files)
                )

                process_file(
                    file_path=file_path,
                    file_index=index,
                    output_path=parquet_dir,
                    conn=conn,
                    chunksize=args.chunksize,
                    overall_stats=overall_stats,
                )

        finally:
            conn.close()

    generate_quality_report(
        output_dir,
        overall_stats
    )

    elapsed = time.perf_counter() - start_time

    LOG.info("Preprocessing completed.")
    LOG.info("Raw records: %s", f"{overall_stats['raw_records']:,}")
    LOG.info("Retained records: %s", f"{overall_stats['retained_records']:,}")
    LOG.info("Elapsed time: %.2f seconds", elapsed)
    LOG.info("Parquet output: %s", parquet_dir)
    LOG.info("Quality reports: %s", output_dir / "quality_reports")


if __name__ == "__main__":
    main()