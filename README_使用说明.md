# NYC 311 数据预处理与四方向数据集生成

## 1. 项目简介

本项目针对 NYC 311 Service Requests 数据集，构建一套可复用、可扩展的数据预处理流程，为后续四个研究方向的数据分析提供统一的数据基础。

整个流程分为两个阶段：

1. **第一阶段：通用数据预处理（General Data Preprocessing）**

   - 读取一个或多个原始 JSONL / JSONL.GZ 文件。
   - 完成通用数据清洗、去重、字段转换和特征工程。
   - 生成统一的标准化 Parquet 数据集 A。
2. **第二阶段：方向数据集生成（Direction Dataset Generation）**

   - 分别从第一阶段生成的统一数据集 A 中读取数据。
   - 根据四个研究方向各自的筛选规则，独立生成四份 Parquet 数据集。
   - 生成详细的数据筛选统计报告。

整体流程如下：

```text
Raw JSONL / JSONL.GZ Files
            |
            v
Stage 1: General Data Preprocessing
            |
            |  Cleaning / Deduplication
            |  Data Transformation
            |  Feature Engineering
            v
     Canonical Dataset A
     (Year-partitioned Parquet)
            |
     +------+------+------+
     |      |      |      |
     v      v      v      v
    B      C      D      E
    |      |      |      |
    v      v      v      v
Complaint Temporal Geospatial Resolution
Analysis  Analysis  Analysis  Analysis
```

**重要说明：** 四个方向的数据集 B、C、D、E 都独立从同一个数据集 A 中筛选生成。它们之间不存在先后依赖关系，也不会因为某个方向排除了一条记录，就影响其他方向的数据。

因此，同一条记录可以同时出现在多个方向的数据集中。四个方向的数据集不是互斥划分，不能简单地将四份数据的记录数相加作为原始数据的唯一记录总数。

---

## 2. 项目文件说明

项目包含三个 Python 文件：

| 文件                               | 作用                                                                        |
| ---------------------------------- | --------------------------------------------------------------------------- |
| `preprocess_nyc311.py`           | 第一阶段：通用数据清洗、转换、去重、特征工程及 Parquet 输出                 |
| `task_preparation.py`            | 定义四个方向的数据筛选规则，以及各方向内部的分析任务准备函数                |
| `generate_direction_datasets.py` | 第二阶段：读取统一 Parquet 数据集，独立生成四个方向的数据集，并生成统计报告 |

三个文件的职责划分：

- `preprocess_nyc311.py` 负责生成统一、标准化的数据源。
- `task_preparation.py` 负责定义各方向的数据处理规则。
- `generate_direction_datasets.py` 负责执行第二阶段的数据读取、筛选、写入和统计。

---

## 3. 环境依赖

建议使用 Python 3.10 或更高版本。

安装依赖：

```bash
python -m pip install "pandas>=2.0" "pyarrow>=12.0"
```

检查安装：

```bash
python -c "import pandas, pyarrow; print('pandas:', pandas.__version__); print('pyarrow:', pyarrow.__version__)"
```

---

## 4. 第一阶段：通用数据预处理

### 4.1 输入数据

第一阶段支持读取一个目录下的多个原始数据文件，包括多个月份或多个年份的数据。

输入数据可以是：

- `.jsonl`
- `.jsonl.gz`

程序会递归搜索输入目录中的数据文件，并逐块读取，避免一次性将所有原始数据加载到内存中。

### 4.2 运行命令

```bash
python preprocess_nyc311.py \
    --input-dir /path/to/raw_data \
    --output-dir /path/to/preprocessing_output \
    --chunksize 100000 \
    --overwrite
```

参数说明：

| 参数             | 说明                               |
| ---------------- | ---------------------------------- |
| `--input-dir`  | 原始 JSONL / JSONL.GZ 文件所在目录 |
| `--output-dir` | 第一阶段结果的输出目录             |
| `--chunksize`  | 每次读取和处理的记录数量           |
| `--overwrite`  | 覆盖已有输出内容                   |

### 4.3 通用清洗规则

第一阶段仅执行适用于所有研究方向的通用清洗，不提前应用四个方向各自的筛选规则。

主要处理内容：

1. **唯一标识检查**

   - 删除缺少 `unique_key` 的记录。
   - 根据 `unique_key` 进行全局去重。
   - 去重范围覆盖本次输入的所有文件，而不只是单个文件。
2. **创建日期检查**

   - 将 `created_date` 转换为统一的日期时间类型。
   - 删除缺失或无法解析 `created_date` 的记录。
3. **字段标准化**

   - 统一日期、数值和分类字段的数据类型。
   - 规范化 Borough 等地理信息字段。
   - 将 ZIP Code 作为字符串处理，保留其原始格式。
4. **地理信息检查**

   - 检查 ZIP Code 是否符合五位数字格式。
   - 检查经纬度是否为有效数值，以及是否处于全球合法坐标范围。
   - 生成地理信息有效性标记。
5. **时间特征工程**

   根据 `created_date` 提取：

   - `year`
   - `month`
   - `year_month`
   - `quarter`
   - `day_of_week`
   - `hour`
   - `is_weekend`
6. **解决情况特征工程**

   根据创建日期和关闭日期生成：

   - `has_closed_date`
   - `is_closed`
   - `is_pending`
   - `closed_before_created`
   - `is_resolution_valid`
   - `resolution_hours`
   - `resolution_days`

   其中，只有存在有效关闭日期且关闭日期不早于创建日期的记录，才计算解决时长。

### 4.4 缺失值和异常记录处理原则

- 不会将 `complaint_type`、`descriptor`、`agency` 等字段的缺失值统一填充为 `"Unknown"`。
- 缺失 `closed_date` 的记录会保留在通用数据集 A 中。
- 关闭日期早于创建日期的记录会保留在通用数据集 A 中，但不会计算有效解决时长，并会被记录在质量报告中。
- 不会因为某个研究方向不需要某条记录，就在第一阶段提前删除它。

### 4.5 第一阶段输出结构

```text
preprocessing_output/
│
├── processed_data/
│   ├── year=2025/
│   │   ├── part-0001-....parquet
│   │   └── ...
│   │
│   └── year=2026/
│       ├── part-0001-....parquet
│       └── ...
│
└── quality_reports/
    ├── README_质量报告.md
    └── overall_report.json
```

`processed_data/` 是后续第二阶段唯一使用的标准化数据源 A。

数据按 `year` 分区存储，方便后续按照年份读取和分析。

`quality_reports/` 保存第一阶段的数据质量统计结果。

---

## 5. 第二阶段：四方向数据集生成

### 5.1 处理逻辑

第二阶段读取第一阶段生成的 Parquet 数据集 A，并分别执行四个方向的筛选规则。

```text
                  Dataset A
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
   Direction 1   Direction 2   Direction 3   Direction 4
        |             |             |             |
        v             v             v             v
   Dataset B     Dataset C     Dataset D     Dataset E
```

四个方向分别读取完整的数据源 A，每个方向拥有独立的筛选逻辑、统计信息和写入缓冲区。

### 5.2 四个方向的筛选规则

#### Direction 1：Complaint Type Analysis（投诉类型分析）

**数据保留规则：**

保留通用数据集 A 中的全部记录，不在方向数据集生成阶段要求 `complaint_type`、`descriptor` 或 `agency` 必须非空。

原因：方向一内部的不同分析任务对字段的要求不同，因此不应该在生成方向数据集时提前删除部分记录。

内部分析任务：

| 分析任务               | 必要字段                           |
| ---------------------- | ---------------------------------- |
| 投诉类型分布           | `complaint_type`                 |
| 投诉细分问题分布       | `complaint_type`、`descriptor` |
| 投诉类型与负责机构关系 | `complaint_type`、`agency`     |

#### Direction 2：Temporal Analysis（时间分析）

**数据保留规则：**

保留 `created_date` 有效的记录。

由于第一阶段已经删除缺少或无法解析创建日期的记录，因此正常情况下，方向二不应该再排除记录。

该方向主要用于：

- 月度、季度和年度投诉趋势。
- 每日及每小时投诉分布。
- 工作日与周末投诉量比较。
- 时间序列变化分析。

#### Direction 3：Geospatial Analysis（地理空间分析）

**数据保留规则：**

只要一条记录至少具有一种有效的地理位置信息，就可以保留：

1. 有效的 NYC Borough。
2. 符合五位数字格式的 ZIP Code。
3. 同时具有有效 Latitude 和 Longitude 的坐标对。

三种条件之间采用 OR 关系，而不是 AND 关系。

即：

```text
Valid Geography =
    Valid Borough
    OR Valid ZIP Code
    OR Valid Coordinate Pair
```

方向三内部可以进一步根据具体任务筛选：

| 分析任务       | 必要字段                           |
| -------------- | ---------------------------------- |
| Borough 分布   | 有效 `borough`                   |
| ZIP Code 分布  | 有效 `incident_zip`              |
| 经纬度空间分析 | 有效 `latitude` 和 `longitude` |

注意：经纬度处于全球合法范围，并不意味着坐标一定位于纽约市行政边界内。如果后续需要严格限制 NYC 范围，还需要使用官方行政边界数据进行空间范围检查。

#### Direction 4：Resolution Analysis（解决情况分析）

**数据保留规则：**

1. `created_date` 必须有效。
2. 如果 `closed_date` 缺失，则保留。
3. 如果 `closed_date` 存在，则不能早于 `created_date`。

逻辑表示：

```text
Valid Resolution Record =
    Valid Created Date
    AND
    (
        Missing Closed Date
        OR Closed Date >= Created Date
    )
```

该方向主要用于：

- 已完成请求的解决时长分析。
- 不同投诉类型的解决效率比较。
- 不同 Borough 或 Agency 的解决效率比较。
- 尚未关闭请求的积压情况分析。

**注意：** `is_closed` 和 `is_pending` 主要根据关闭日期判断，而不是简单根据 `status` 字段判断。对于无法解析的非空关闭日期，应在质量检查中单独识别，不能直接将其视为真正缺失的关闭日期。

---

## 6. 第二阶段的 Batch 读取与 Parquet 写入机制

为了控制内存使用，并避免产生大量过小的 Parquet 文件，第二阶段采用两个独立的参数：

| 参数                  |  默认值 | 作用                                              |
| --------------------- | ------: | ------------------------------------------------- |
| `batch_size`        | 100,000 | 控制 Scanner 每次读取数据的目标 Batch 大小        |
| `write_buffer_size` | 100,000 | 控制每个方向累计多少条记录后写入一个 Parquet 文件 |

运行时需要注意：

- Scanner 实际返回的 Batch 大小可能小于设置的 `batch_size`。
- 一个方向的数据可能来自多个 Scanner Batch。
- 每个方向拥有独立的写入缓冲区。
- 当某个方向的缓冲区达到 `write_buffer_size` 时，程序写入一个 Parquet 文件。
- 处理结束后，剩余记录也会写入最后一个 Parquet 文件。

例如，某方向最终保留 348,180 条记录，且写入缓冲区为 100,000，则预计生成：

```text
part-0001.parquet    100,000 rows
part-0002.parquet    100,000 rows
part-0003.parquet    100,000 rows
part-0004.parquet     48,180 rows
```

因此，Scanner Batch 数量不等于输出 Parquet 文件数量。

---

## 7. 第二阶段运行命令

```bash
python generate_direction_datasets.py \
    --parquet-dir /path/to/preprocessing_output/processed_data \
    --output-dir /path/to/direction_outputs \
    --direction all \
    --batch-size 100000 \
    --write-buffer-size 100000 \
    --overwrite
```

参数说明：

| 参数                    | 说明                                |
| ----------------------- | ----------------------------------- |
| `--parquet-dir`       | 第一阶段生成的统一 Parquet 数据集 A |
| `--output-dir`        | 第二阶段四个方向数据集的输出目录    |
| `--direction all`     | 生成全部四个方向                    |
| `--direction 1`       | 只生成方向一                        |
| `--direction 2`       | 只生成方向二                        |
| `--direction 3`       | 只生成方向三                        |
| `--direction 4`       | 只生成方向四                        |
| `--batch-size`        | Scanner 读取 Batch 的目标大小       |
| `--write-buffer-size` | 每个方向的 Parquet 写入缓冲区大小   |
| `--overwrite`         | 覆盖本次选择的方向输出目录          |

如果只想重新生成方向三，可以执行：

```bash
python generate_direction_datasets.py \
    --parquet-dir /path/to/preprocessing_output/processed_data \
    --output-dir /path/to/direction_outputs \
    --direction 3 \
    --batch-size 100000 \
    --write-buffer-size 100000 \
    --overwrite
```

`--overwrite` 只作用于本次选择的方向输出，不会修改第一阶段生成的通用数据集 A。

---

## 8. 第二阶段输出结构

```text
direction_outputs/
│
├── direction_1_complaint/
│   ├── part-0001.parquet
│   ├── part-0002.parquet
│   └── ...
│
├── direction_2_temporal/
│   ├── part-0001.parquet
│   └── ...
│
├── direction_3_geospatial/
│   ├── part-0001.parquet
│   └── ...
│
├── direction_4_resolution/
│   ├── part-0001.parquet
│   └── ...
│
├── direction_dataset_summary.json
└── direction_dataset_report.md
```

其中：

- 四个目录分别存储四个方向独立筛选后的 Parquet 数据集。
- `direction_dataset_summary.json` 保存结构化统计信息，方便后续程序读取。
- `direction_dataset_report.md` 保存便于阅读的 Markdown 数据处理报告。

---

## 9. 数据处理报告说明

第二阶段会记录每个方向的数据处理情况。

主要统计内容包括：

1. **总体处理信息**

   - 输入数据总记录数。
   - Scanner Batch 数量。
   - 实际使用的 Batch 和写入缓冲区参数。
   - 每个方向生成的 Parquet 文件数量。
2. **方向级数据筛选统计**

   - 输入记录数。
   - 保留记录数。
   - 排除记录数。
   - 数据保留率和排除率。
3. **排除原因统计**

   - 记录各方向因不符合筛选规则而被排除的数量。
4. **字段质量统计**

   - 关键字段缺失情况。
   - 日期、地理位置等字段的有效性。
5. **任务级分析资格统计**

   - 方向一不同分析任务的可用记录数量。
   - 方向三不同地理分析任务的可用记录数量。
   - 方向四已完成和待处理记录的相关统计。

需要注意：四个方向的数据集之间可能存在记录重叠，因此不能将四个方向的保留记录数相加，作为唯一记录总数。

---

## 10. 重要设计约定

1. 第一阶段只负责通用清洗，不提前执行四个方向的专属筛选。
2. 第一阶段对所有输入文件进行全局 `unique_key` 去重。
3. 第二阶段四个方向均从完整数据集 A 独立读取。
4. 四个方向之间允许存在相同的 `unique_key`。
5. 方向一不在数据集生成阶段要求投诉类型、描述或机构字段非空。
6. 方向二主要依赖有效的 `created_date`。
7. 方向三采用 Borough、ZIP Code、经纬度三种地理信息的 OR 筛选规则。
8. 方向四允许缺少关闭日期的记录保留，但排除关闭日期早于创建日期的记录。
9. 第二阶段的读取 Batch 和 Parquet 写入缓冲区是两个独立参数。
10. 第一阶段的通用数据集 A 应始终保持独立，不因第二阶段某个方向的筛选而发生变化。

---

## 11. 完整执行流程

从原始数据到四个方向数据集，完整执行顺序如下：

**Step 1：准备原始数据**

将一个或多个 NYC 311 JSONL / JSONL.GZ 文件放入原始数据目录。

**Step 2：执行第一阶段**

运行 `preprocess_nyc311.py`，生成通用数据集 A 和质量报告。

**Step 3：检查第一阶段结果**

检查 `processed_data/` 是否存在，以及 `overall_report.json` 和 `README_质量报告.md` 是否正常生成。

**Step 4：执行第二阶段**

运行 `generate_direction_datasets.py`，从 A 中独立生成四个方向的数据集。

**Step 5：检查第二阶段结果**

检查四个方向的 Parquet 文件，以及 `direction_dataset_summary.json` 和 `direction_dataset_report.md`。

**Step 6：开展后续分析**

分别读取四个方向的数据集，开展投诉类型、时间趋势、地理空间和解决情况分析，并为后续 Hadoop MapReduce、Apache Spark 等并行计算任务提供数据输入。
