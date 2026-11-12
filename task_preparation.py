# task_preparation.py

from __future__ import annotations

import pandas as pd


# ============================================================
# 1. 通用辅助函数
# ============================================================

VALID_BOROUGHS = {
    "MANHATTAN",
    "BROOKLYN",
    "QUEENS",
    "BRONX",
    "STATEN ISLAND",
}


def is_non_missing(series: pd.Series) -> pd.Series:
    """
    判断字段是否非空。
    同时将空字符串视为缺失。
    """
    return (
        series.notna()
        & series.astype("string").str.strip().ne("").fillna(False)
    )


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """
    将日期字段统一转换为 UTC datetime。
    无法解析的值转换为 NaT。
    """
    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True
    )


def get_datetime_columns(df: pd.DataFrame):
    """
    获取创建日期和关闭日期。
    """
    if "created_date" in df.columns:
        created = parse_datetime_series(df["created_date"])
    else:
        created = pd.Series(
            pd.NaT,
            index=df.index,
            dtype="datetime64[ns, UTC]"
        )

    if "closed_date" in df.columns:
        closed = parse_datetime_series(df["closed_date"])
    else:
        closed = pd.Series(
            pd.NaT,
            index=df.index,
            dtype="datetime64[ns, UTC]"
        )

    return created, closed


def get_borough_valid_mask(df: pd.DataFrame) -> pd.Series:
    """
    Borough 有效性：
    必须属于 NYC 五个行政区。
    """
    if "borough" not in df.columns:
        return pd.Series(False, index=df.index)

    borough = (
        df["borough"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return borough.isin(VALID_BOROUGHS).fillna(False)


def get_zip_valid_mask(df: pd.DataFrame) -> pd.Series:
    """
    ZIP 有效性：
    必须是五位数字。

    兼容部分数值型 ZIP 被转换为 10001.0 的情况。
    """
    if "incident_zip" not in df.columns:
        return pd.Series(False, index=df.index)

    zip_code = (
        df["incident_zip"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    return zip_code.str.fullmatch(r"\d{5}", na=False)


def get_coordinate_valid_mask(df: pd.DataFrame) -> pd.Series:
    """
    坐标有效性：
    必须同时具有有效 latitude 和 longitude，
    且处于全球合法经纬度范围。

    注意：这不代表坐标一定在 NYC 边界内。
    """
    if "latitude" not in df.columns:
        latitude = pd.Series(float("nan"), index=df.index)
    else:
        latitude = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )

    if "longitude" not in df.columns:
        longitude = pd.Series(float("nan"), index=df.index)
    else:
        longitude = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )

    return (
        latitude.notna()
        & longitude.notna()
        & latitude.between(-90, 90)
        & longitude.between(-180, 180)
    )


def get_geography_masks(df: pd.DataFrame):
    """
    返回三种地理信息的有效性掩码。
    """
    borough_valid = get_borough_valid_mask(df)
    zip_valid = get_zip_valid_mask(df)
    coordinates_valid = get_coordinate_valid_mask(df)

    return borough_valid, zip_valid, coordinates_valid


# ============================================================
# 2. 四个方向的独立筛选函数
# ============================================================

def prepare_direction_1_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    方向一：Complaint Type Analysis

    不进行方向级删除。
    缺少 complaint_type、descriptor、agency 的记录仍然保留。
    """
    return df.copy()


def prepare_direction_2_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    方向二：Temporal Analysis

    只保留 created_date 有效的记录。
    """
    created, _ = get_datetime_columns(df)

    mask = created.notna()

    return df.loc[mask].copy()


def prepare_direction_3_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    方向三：Geospatial Analysis

    Borough、ZIP、坐标三者至少一个有效即可保留。
    """
    borough_valid, zip_valid, coordinates_valid = (
        get_geography_masks(df)
    )

    mask = (
        borough_valid
        | zip_valid
        | coordinates_valid
    )

    return df.loc[mask].copy()


def prepare_direction_4_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    方向四：Resolution Analysis

    保留条件：
    1. created_date 有效；
    2. closed_date 可以为空；
    3. 如果 closed_date 存在，则不能早于 created_date。

    没有 closed_date 的记录作为 Pending 候选记录保留。
    """
    created, closed = get_datetime_columns(df)

    valid_created = created.notna()

    closed_before_created = (
        closed.notna()
        & valid_created
        & (closed < created)
    )

    mask = (
        valid_created
        & ~closed_before_created
    )

    return df.loc[mask].copy()


# ============================================================
# 3. 四个方向的函数映射
# ============================================================

DIRECTION_BUILDERS = {
    1: prepare_direction_1_dataset,
    2: prepare_direction_2_dataset,
    3: prepare_direction_3_dataset,
    4: prepare_direction_4_dataset,
}


DIRECTION_NAMES = {
    1: "complaint_type",
    2: "temporal",
    3: "geospatial",
    4: "resolution",
}


DIRECTION_DESCRIPTIONS = {
    1: "投诉类型分析（Complaint Type Analysis）",
    2: "时间分析（Temporal Analysis）",
    3: "地理空间分析（Geospatial Analysis）",
    4: "解决情况分析（Resolution Analysis）",
}


# ============================================================
# 4. 分析任务级资格判断
# ============================================================

def get_direction_1_task_masks(
    df: pd.DataFrame
) -> dict[str, pd.Series]:
    """
    方向一内部不同分析任务的数据资格。

    这些条件只用于统计分析资格，不用于方向一数据集筛选。
    """
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

    return {
        "complaint_type_analysis": complaint_valid,
        "descriptor_analysis": (
            complaint_valid & descriptor_valid
        ),
        "agency_analysis": (
            complaint_valid & agency_valid
        ),
    }


def get_direction_3_task_masks(
    df: pd.DataFrame
) -> dict[str, pd.Series]:
    """
    方向三内部不同地理分析任务的数据资格。
    """
    borough_valid, zip_valid, coordinates_valid = (
        get_geography_masks(df)
    )

    return {
        "borough_analysis": borough_valid,
        "zip_analysis": zip_valid,
        "coordinate_analysis": coordinates_valid,
    }