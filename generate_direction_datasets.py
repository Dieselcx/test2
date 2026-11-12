# generate_direction_datasets.py

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from task_preparation import (
    DIRECTION_BUILDERS,
    DIRECTION_NAMES,
    DIRECTION_DESCRIPTIONS,
    get_datetime_columns,
    get_geography_masks,
    get_direction_1_task_masks,
    get_direction_3_task_masks,
    is_non_missing,
)


# ============================================================
# 1. 基本配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100_000
DEFAULT_WRITE_BUFFER_SIZE = 100_000

DIRECTION_FOLDER_NAMES = {
    1: "direction_1_complaint_type",
    2: "direction_2_temporal",
    3: "direction_3_geospatial",
    4: "direction_4_resolution",
}


# ============================================================
# 2. 统计结构
# ============================================================

def create_direction_stats(direction: int) -> dict:
    """
    创建每个方向独立的统计结构。
    """
    stats = {
        "direction": direction,
        "name": DIRECTION_NAMES[direction],
        "description": DIRECTION_DESCRIPTIONS[direction],

        "input_rows": 0,
        "retained_rows": 0,
        "excluded_rows": 0,

        "output_files": 0,

        "exclusion_reasons": {},
        "field_quality": {},
        "task_eligibility": {},
        "geography_combinations": {},
    }

    if direction == 1:
        stats["field_quality"] = {
            "missing_complaint_type": 0,
            "missing_descriptor": 0,
            "missing_agency": 0,
        }

        stats["task_eligibility"] = {
            "complaint_type_analysis": 0,
            "descriptor_analysis": 0,
            "agency_analysis": 0,
        }

    elif direction == 2:
        stats["field_quality"] = {
            "invalid_or_missing_created_date": 0,
        }

        stats["exclusion_reasons"] = {
            "invalid_or_missing_created_date": 0,
        }

    elif direction == 3:
        stats["field_quality"] = {
            "valid_borough": 0,
            "invalid_or_missing_borough": 0,

            "valid_zip": 0,
            "invalid_or_missing_zip": 0,

            "valid_coordinate_pair": 0,
            "invalid_or_missing_coordinate_pair": 0,
        }

        stats["exclusion_reasons"] = {
            "no_valid_geography_information": 0,
        }

        stats["task_eligibility"] = {
            "borough_analysis": 0,
            "zip_analysis": 0,
            "coordinate_analysis": 0,
        }

        stats["geography_combinations"] = {
            "borough_only": 0,
            "zip_only": 0,
            "coordinates_only": 0,
            "borough_and_zip": 0,
            "borough_and_coordinates": 0,
            "zip_and_coordinates": 0,
            "all_three": 0,
            "none": 0,
        }

    elif direction == 4:
        stats["field_quality"] = {
            "invalid_or_missing_created_date": 0,
            "missing_closed_date": 0,
            "valid_closed_date": 0,
            "closed_before_created": 0,
        }

        stats["exclusion_reasons"] = {
            "invalid_or_missing_created_date": 0,
            "closed_before_created": 0,
        }

        stats["task_eligibility"] = {
            "completed_records": 0,
            "pending_candidates": 0,
        }

    return stats


def safe_count(mask: pd.Series) -> int:
    """
    将布尔掩码转换为整数数量。
    """
    return int(mask.fillna(False).sum())


# ============================================================
# 3. 更新方向统计
# ============================================================

def update_direction_stats(
    direction: int,
    df: pd.DataFrame,
    retained_df: pd.DataFrame,
    stats: dict,
) -> None:
    """
    使用当前原始 Batch 更新指定方向的统计信息。

    注意：
    每个方向都以同一个原始 df 为输入，
    不会使用其他方向筛选后的结果。
    """
    input_rows = len(df)
    retained_rows = len(retained_df)

    stats["input_rows"] += input_rows
    stats["retained_rows"] += retained_rows
    stats["excluded_rows"] += input_rows - retained_rows

    # --------------------------------------------------------
    # 方向 1：Complaint Type
    # --------------------------------------------------------

    if direction == 1:
        complaint_valid = (
            is_non_missing(df["complaint_type"])
            if "complaint_type" in df.columns
            else pd.Series(False, index=df.index)
        )

        descriptor_valid = (
            is_non_missing(df["descriptor"])
            if "descriptor" in df.columns
            else pd.Series(False, index=df.index)
        )

        agency_valid = (
            is_non_missing(df["agency"])
            if "agency" in df.columns
            else pd.Series(False, index=df.index)
        )

        stats["field_quality"]["missing_complaint_type"] += (
            input_rows - safe_count(complaint_valid)
        )

        stats["field_quality"]["missing_descriptor"] += (
            input_rows - safe_count(descriptor_valid)
        )

        stats["field_quality"]["missing_agency"] += (
            input_rows - safe_count(agency_valid)
        )

        task_masks = get_direction_1_task_masks(df)

        for task_name, mask in task_masks.items():
            stats["task_eligibility"][task_name] += safe_count(mask)

    # --------------------------------------------------------
    # 方向 2：Temporal
    # --------------------------------------------------------

    elif direction == 2:
        created, _ = get_datetime_columns(df)

        invalid_created = created.isna()

        invalid_count = safe_count(invalid_created)

        stats["field_quality"][
            "invalid_or_missing_created_date"
        ] += invalid_count

        stats["exclusion_reasons"][
            "invalid_or_missing_created_date"
        ] += invalid_count

    # --------------------------------------------------------
    # 方向 3：Geospatial
    # --------------------------------------------------------

    elif direction == 3:
        borough_valid, zip_valid, coordinates_valid = (
            get_geography_masks(df)
        )

        stats["field_quality"]["valid_borough"] += (
            safe_count(borough_valid)
        )

        stats["field_quality"]["invalid_or_missing_borough"] += (
            input_rows - safe_count(borough_valid)
        )

        stats["field_quality"]["valid_zip"] += (
            safe_count(zip_valid)
        )

        stats["field_quality"]["invalid_or_missing_zip"] += (
            input_rows - safe_count(zip_valid)
        )

        stats["field_quality"]["valid_coordinate_pair"] += (
            safe_count(coordinates_valid)
        )

        stats["field_quality"][
            "invalid_or_missing_coordinate_pair"
        ] += (
            input_rows - safe_count(coordinates_valid)
        )

        geography_masks = {
            "borough": borough_valid,
            "zip": zip_valid,
            "coordinates": coordinates_valid,
        }

        b = borough_valid
        z = zip_valid
        c = coordinates_valid

        combination_masks = {
            "borough_only": b & ~z & ~c,
            "zip_only": ~b & z & ~c,
            "coordinates_only": ~b & ~z & c,
            "borough_and_zip": b & z & ~c,
            "borough_and_coordinates": b & ~z & c,
            "zip_and_coordinates": ~b & z & c,
            "all_three": b & z & c,
            "none": ~b & ~z & ~c,
        }

        for name, mask in combination_masks.items():
            stats["geography_combinations"][name] += safe_count(mask)

        no_geography = combination_masks["none"]

        stats["exclusion_reasons"][
            "no_valid_geography_information"
        ] += safe_count(no_geography)

        task_masks = get_direction_3_task_masks(df)

        for task_name, mask in task_masks.items():
            stats["task_eligibility"][task_name] += safe_count(mask)

    # --------------------------------------------------------
    # 方向 4：Resolution
    # --------------------------------------------------------

    elif direction == 4:
        created, closed = get_datetime_columns(df)

        invalid_created = created.isna()

        closed_missing = closed.isna()

        closed_valid = closed.notna()

        # 只有创建日期有效时，才判断关闭日期是否早于创建日期。
        closed_before_created = (
            created.notna()
            & closed.notna()
            & (closed < created)
        )

        invalid_created_count = safe_count(invalid_created)
        closed_before_count = safe_count(closed_before_created)

        # Pending 候选记录：创建日期有效，但没有有效关闭日期。
        pending_candidates = (
            created.notna()
            & closed.isna()
        )

        # Completed：创建日期有效，关闭日期有效，
        # 且关闭日期不早于创建日期。
        completed_records = (
            created.notna()
            & closed.notna()
            & (closed >= created)
        )

        stats["field_quality"][
            "invalid_or_missing_created_date"
        ] += invalid_created_count

        stats["field_quality"]["missing_closed_date"] += (
            safe_count(closed_missing)
        )

        stats["field_quality"]["valid_closed_date"] += (
            safe_count(closed_valid)
        )

        stats["field_quality"]["closed_before_created"] += (
            closed_before_count
        )

        stats["exclusion_reasons"][
            "invalid_or_missing_created_date"
        ] += invalid_created_count

        stats["exclusion_reasons"][
            "closed_before_created"
        ] += closed_before_count

        stats["task_eligibility"]["completed_records"] += (
            safe_count(completed_records)
        )

        stats["task_eligibility"]["pending_candidates"] += (
            safe_count(pending_candidates)
        )


# ============================================================
# 4. Parquet 写入缓冲区
# ============================================================

def write_parquet_file(
    df: pd.DataFrame,
    direction_dir: Path,
    file_number: int,
) -> Path:
    """
    将一个完整缓冲区写入一个 Parquet 文件。
    """
    direction_dir.mkdir(parents=True, exist_ok=True)

    file_path = direction_dir / (
        f"part-{file_number:06d}.parquet"
    )

    df.to_parquet(
        file_path,
        engine="pyarrow",
        index=False,
        compression="snappy",
    )

    return file_path


def append_to_buffer(
    direction: int,
    new_df: pd.DataFrame,
    buffers: dict,
    buffer_size: int,
    direction_dir: Path,
    file_numbers: dict,
    stats: dict,
) -> None:
    """
    将当前方向筛选后的数据放入独立缓冲区。

    达到 buffer_size 后，按固定行数切分并写入。
    因此不会因为 Scanner 返回 32,768 行一个 Batch，
    就产生 32,768 行一个 Parquet 文件。

    每个方向都有自己的 buffer。
    """
    if new_df.empty:
        return

    if buffers[direction] is None:
        buffers[direction] = new_df.reset_index(drop=True).copy()
    else:
        buffers[direction] = pd.concat(
            [
                buffers[direction],
                new_df.reset_index(drop=True),
            ],
            ignore_index=True,
        )

    buffer_df = buffers[direction]

    # 如果累计行数达到写入阈值，则按阈值写入。
    while len(buffer_df) >= buffer_size:
        output_df = buffer_df.iloc[:buffer_size].copy()

        file_numbers[direction] += 1

        write_parquet_file(
            output_df,
            direction_dir,
            file_numbers[direction],
        )

        stats[direction]["output_files"] += 1

        buffer_df = (
            buffer_df.iloc[buffer_size:]
            .reset_index(drop=True)
        )

    buffers[direction] = buffer_df


def flush_remaining_buffers(
    directions: list[int],
    buffers: dict,
    output_dir: Path,
    file_numbers: dict,
    stats: dict,
) -> None:
    """
    扫描结束后，将每个方向不足一个缓冲区大小的剩余数据写入。
    """
    for direction in directions:
        remaining_df = buffers[direction]

        if remaining_df is None or remaining_df.empty:
            continue

        file_numbers[direction] += 1

        direction_dir = (
            output_dir / DIRECTION_FOLDER_NAMES[direction]
        )

        write_parquet_file(
            remaining_df,
            direction_dir,
            file_numbers[direction],
        )

        stats[direction]["output_files"] += 1

        buffers[direction] = None


# ============================================================
# 5. JSON 数值与报告辅助函数
# ============================================================

def calculate_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return round(numerator / denominator * 100, 4)


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )


def format_number(value: int) -> str:
    return f"{value:,}"


def format_percentage(value: float) -> str:
    return f"{value:.2f}%"


# ============================================================
# 6. Markdown 报告生成
# ============================================================

def generate_markdown_report(
    summary: dict,
    output_path: Path,
) -> None:
    """
    生成详细的第二阶段 Markdown 报告。
    """
    lines = []

    def add(text: str = ""):
        lines.append(text)

    add("# 第二阶段方向数据集生成报告")
    add()

    add("## 1. 总体处理概况")
    add()

    add(f"- 生成时间：{summary['generation_time']}")
    add(f"- 输入数据集：`{summary['input_dataset']}`")
    add(f"- 输入记录总数：{format_number(summary['input_rows'])}")
    add(f"- 处理方向数量：{summary['direction_count']}")
    add(f"- Scanner Batch 数量：{summary['scanner_batch_count']}")
    add(f"- Scanner Batch Size：{format_number(summary['batch_size'])}")
    add(
        "- Parquet 写入缓冲区大小："
        f"{format_number(summary['write_buffer_size'])}"
    )
    add(f"- 输出 Parquet 文件总数：{summary['total_output_files']}")
    add()

    add("### 各方向输出概况")
    add()
    add("| 方向 | 输入记录 | 保留记录 | 排除记录 | 保留率 | Parquet 文件数 |")
    add("|---|---:|---:|---:|---:|---:|")

    for direction in summary["directions"]:
        add(
            f"| {direction['description']} "
            f"| {format_number(direction['input_rows'])} "
            f"| {format_number(direction['retained_rows'])} "
            f"| {format_number(direction['excluded_rows'])} "
            f"| {format_percentage(direction['retention_rate'])} "
            f"| {direction['output_files']} |"
        )

    add()
    add("> **重要说明：** 四个方向分别从同一个通用数据集 A 独立筛选，")
    add("> 各方向之间允许存在相同的 `unique_key`。因此，不能将四个方向的")
    add("> 保留记录数量相加后，当作全局去重后的唯一记录数量。")
    add()

    add("## 2. 各方向详细统计")
    add()

    for direction in summary["directions"]:
        number = direction["direction"]

        add(f"### 2.{number} {direction['description']}")
        add()

        add("**筛选规则**")
        add()

        if number == 1:
            add(
                "- 不进行方向级删除，保留通用数据集 A 中的全部记录。"
            )
            add(
                "- 缺少 `complaint_type`、`descriptor` 或 `agency` "
                "只影响相应分析任务的可用记录数量，不影响方向一数据集保留。"
            )

        elif number == 2:
            add(
                "- 仅保留 `created_date` 有效的记录。"
            )

        elif number == 3:
            add(
                "- Borough、五位 ZIP、有效经纬度坐标三者中，"
                "至少有一种有效即可保留。"
            )
            add(
                "- 只有三种地理信息全部无效或缺失时才排除。"
            )
            add(
                "- 坐标有效仅代表经纬度数值处于全球合法范围，"
                "不代表坐标一定位于 NYC 边界内。"
            )

        elif number == 4:
            add(
                "- `created_date` 必须有效。"
            )
            add(
                "- `closed_date` 可以为空；为空的记录作为 Pending 候选记录保留。"
            )
            add(
                "- 如果 `closed_date` 存在，则不能早于 `created_date`。"
            )
            add(
                "- 是否完成以有效日期字段判断，不直接使用 `status` "
                "代替关闭日期判断。"
            )

        add()
        add("**处理结果**")
        add()
        add(f"- 输入记录：{format_number(direction['input_rows'])}")
        add(f"- 保留记录：{format_number(direction['retained_rows'])}")
        add(f"- 排除记录：{format_number(direction['excluded_rows'])}")
        add(
            "- 保留率："
            f"{format_percentage(direction['retention_rate'])}"
        )
        add(
            "- 排除率："
            f"{format_percentage(direction['exclusion_rate'])}"
        )
        add(f"- 输出 Parquet 文件：{direction['output_files']}")
        add()

        # 排除原因
        add("**排除原因统计**")
        add()

        if direction["exclusion_reasons"]:
            add("| 排除原因 | 记录数量 |")
            add("|---|---:|")

            for reason, count in direction["exclusion_reasons"].items():
                add(f"| `{reason}` | {format_number(count)} |")
        else:
            add("该方向没有方向级排除条件，因此没有记录被方向级删除。")

        add()

        # 字段质量
        if direction["field_quality"]:
            add("**字段质量统计**")
            add()
            add("| 指标 | 记录数量 |")
            add("|---|---:|")

            for name, count in direction["field_quality"].items():
                add(f"| `{name}` | {format_number(count)} |")

            add()

        # 任务级资格
        if direction["task_eligibility"]:
            add("**分析任务的数据资格统计**")
            add()
            add("| 分析任务 | 可用记录数量 |")
            add("|---|---:|")

            for name, count in direction["task_eligibility"].items():
                add(f"| `{name}` | {format_number(count)} |")

            add()

        # 地理组合
        if direction["geography_combinations"]:
            add("**地理信息组合统计**")
            add()
            add("| 地理信息组合 | 记录数量 |")
            add("|---|---:|")

            for name, count in direction["geography_combinations"].items():
                add(f"| `{name}` | {format_number(count)} |")

            add()

    add("## 3. 统计口径说明")
    add()
    add(
        "1. 每个方向的输入记录数均以完整通用数据集 A 为基础，"
        "不是以前一个方向的输出作为下一个方向的输入。"
    )
    add(
        "2. 各方向的输入记录数应相同；方向之间的输出记录可以重叠。"
    )
    add(
        "3. 字段缺失统计可能相互重叠，不应将多个字段缺失数量相加，"
        "并将结果解释为实际排除记录数。"
    )
    add(
        "4. 方向级排除记录数以输入记录数减去保留记录数计算。"
    )
    add(
        "5. 方向三的地理信息组合类别互斥，八种组合的数量之和"
        "应等于该方向的输入记录数。"
    )
    add(
        "6. 方向四的排除原因按创建日期无效、关闭日期早于创建日期"
        "进行互斥统计。"
    )
    add(
        "7. `pending_candidates` 指创建日期有效且没有有效关闭日期的记录，"
        "并不意味着这些记录的 `status` 一定为 Pending。"
    )
    add(
        "8. 本报告的方向级统计来自第二阶段实际扫描的数据，"
        "不改变第一阶段通用数据集 A。"
    )
    add()

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# 7. 主处理流程
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "从通用 NYC 311 Parquet 数据集 A 独立生成四个方向的数据集。"
        )
    )

    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="第一阶段生成的通用 Parquet 数据集目录。",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="第二阶段输出目录。",
    )

    parser.add_argument(
        "--direction",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="需要生成的方向，默认为 all。",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Scanner 读取批次大小，默认 100000。",
    )

    parser.add_argument(
        "--write-buffer-size",
        type=int,
        default=DEFAULT_WRITE_BUFFER_SIZE,
        help="每个方向累计多少条记录后写出一个 Parquet 文件，默认 100000。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖本次选择的方向目录。",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_dir = Path(args.parquet_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"输入目录不存在：{input_dir}"
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0。")

    if args.write_buffer_size <= 0:
        raise ValueError("--write-buffer-size 必须大于 0。")

    if args.direction == "all":
        directions = [1, 2, 3, 4]
    else:
        directions = [int(args.direction)]

    # --------------------------------------------------------
    # 检查输出目录
    # --------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    selected_output_dirs = [
        output_dir / DIRECTION_FOLDER_NAMES[d]
        for d in directions
    ]

    for direction_dir in selected_output_dirs:
        if direction_dir.exists():
            has_files = any(direction_dir.iterdir())

            if has_files and not args.overwrite:
                raise FileExistsError(
                    f"输出目录已存在且非空：{direction_dir}\n"
                    "如需重新生成，请添加 --overwrite。"
                )

            if args.overwrite:
                shutil.rmtree(direction_dir)

        direction_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 初始化统计和缓冲区
    # --------------------------------------------------------

    stats = {
        direction: create_direction_stats(direction)
        for direction in directions
    }

    buffers = {
        direction: None
        for direction in directions
    }

    file_numbers = {
        direction: 0
        for direction in directions
    }

    scanner_batch_count = 0
    total_input_rows = 0

    # --------------------------------------------------------
    # 加载通用数据集 A
    # --------------------------------------------------------

    LOGGER.info("正在加载通用数据集 A：%s", input_dir)

    dataset = ds.dataset(
        str(input_dir),
        format="parquet",
        partitioning="hive",
    )

    scanner = dataset.scanner(
        batch_size=args.batch_size,
    )

    # --------------------------------------------------------
    # 扫描 A
    # --------------------------------------------------------

    LOGGER.info(
        "开始第二阶段处理：directions=%s, batch_size=%s, "
        "write_buffer_size=%s",
        directions,
        args.batch_size,
        args.write_buffer_size,
    )

    for batch in scanner.to_batches():
        scanner_batch_count += 1

        df = batch.to_pandas()

        batch_rows = len(df)
        total_input_rows += batch_rows

        LOGGER.info(
            f"Batch {scanner_batch_count}：读取 {batch_rows:,} 条记录"
            )

        # 所有方向都使用同一个原始 df。
        # 不允许将某个方向筛选后的结果传给另一个方向。
        for direction in directions:
            builder = DIRECTION_BUILDERS[direction]

            # Builder 内部返回独立 DataFrame。
            direction_df = builder(df)

            # 使用原始 Batch 更新该方向的统计。
            update_direction_stats(
                direction=direction,
                df=df,
                retained_df=direction_df,
                stats=stats[direction],
            )

            direction_dir = (
                output_dir / DIRECTION_FOLDER_NAMES[direction]
            )

            # 每个方向使用自己的写入缓冲区。
            append_to_buffer(
                direction=direction,
                new_df=direction_df,
                buffers=buffers,
                buffer_size=args.write_buffer_size,
                direction_dir=direction_dir,
                file_numbers=file_numbers,
                stats=stats,
            )

    # --------------------------------------------------------
    # 写入每个方向剩余的数据
    # --------------------------------------------------------

    flush_remaining_buffers(
        directions=directions,
        buffers=buffers,
        output_dir=output_dir,
        file_numbers=file_numbers,
        stats=stats,
    )

    # --------------------------------------------------------
    # 整理统计结果
    # --------------------------------------------------------

    generation_time = datetime.now(
        timezone.utc
    ).astimezone().isoformat(timespec="seconds")

    direction_results = []

    for direction in directions:
        item = stats[direction]

        input_rows = item["input_rows"]
        retained_rows = item["retained_rows"]
        excluded_rows = item["excluded_rows"]

        item["retention_rate"] = calculate_rate(
            retained_rows,
            input_rows,
        )

        item["exclusion_rate"] = calculate_rate(
            excluded_rows,
            input_rows,
        )

        # 检查数量是否能够对账。
        if input_rows != retained_rows + excluded_rows:
            raise RuntimeError(
                f"方向 {direction} 的输入、保留和排除数量不一致。"
            )

        direction_results.append(item)

    total_output_files = sum(
        item["output_files"]
        for item in direction_results
    )

    # --------------------------------------------------------
    # 总结数据
    # --------------------------------------------------------

    summary = {
        "generation_time": generation_time,

        "input_dataset": str(input_dir),
        "output_directory": str(output_dir),

        "input_rows": total_input_rows,

        "direction_count": len(directions),
        "directions_selected": directions,

        "scanner_batch_count": scanner_batch_count,

        "batch_size": args.batch_size,
        "write_buffer_size": args.write_buffer_size,

        "total_output_files": total_output_files,

        "important_note": (
            "四个方向分别从同一个通用数据集 A 独立生成，"
            "方向之间允许存在相同 unique_key，"
            "各方向输出行数不能直接相加作为全局唯一记录数。"
        ),

        "directions": direction_results,
    }

    # --------------------------------------------------------
    # 保存 JSON 和 Markdown 报告
    # --------------------------------------------------------

    json_path = output_dir / "direction_dataset_summary.json"

    markdown_path = output_dir / "direction_dataset_report.md"

    save_json(summary, json_path)

    generate_markdown_report(
        summary=summary,
        output_path=markdown_path,
    )

    # --------------------------------------------------------
    # 打印最终结果
    # --------------------------------------------------------

    LOGGER.info("=" * 60)
    LOGGER.info("第二阶段处理完成")
    LOGGER.info("=" * 60)

    LOGGER.info(f"输入记录总数：{total_input_rows:,}")
    LOGGER.info("Scanner Batch 数量：%d", scanner_batch_count)
    LOGGER.info("输出 Parquet 文件总数：%d", total_output_files)

    for item in direction_results:
        LOGGER.info(
            f"{item['description']}："
            f"输入 {item['input_rows']:,}，"
            f"保留 {item['retained_rows']:,}，"
            f"排除 {item['excluded_rows']:,}，"
            f"文件 {item['output_files']}"
            )

    LOGGER.info("JSON 报告：%s", json_path)
    LOGGER.info("Markdown 报告：%s", markdown_path)


if __name__ == "__main__":
    main()