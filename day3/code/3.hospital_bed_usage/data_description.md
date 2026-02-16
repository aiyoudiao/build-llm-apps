# 医院床位使用数据字段说明文档

本文档详细说明了 `hospital_bed_usage_data.xlsx` 数据表中各字段的含义、数据类型及用途。

## 数据表结构

该数据表包含 13 个字段，记录了不同医院、不同科室及病房的床位使用情况。

| 字段名 (Column Name) | 数据类型 (Type) | 中文含义 | 说明与示例 |
| :--- | :--- | :--- | :--- |
| **hospital_id** | String | 医院ID | 医院的唯一标识符，例如 `HH001`。 |
| **hospital_name** | String | 医院名称 | 医院的全称，例如 `玛丽医院`。 |
| **hospital_district** | String | 医院区域 | 医院所在的行政区域或地理分区，例如 `港岛`。 |
| **department_id** | String | 科室ID | 医疗科室的唯一标识符，用于区分不同科室。 |
| **department_name** | String | 科室名称 | 科室的具体名称，例如 `内科`、`外科`。 |
| **ward_id** | String | 病房ID | 病房的唯一标识符，用于区分同一科室下的不同病房。 |
| **ward_name** | String | 病房名称 | 病房的名称或编号，例如 `A101`。 |
| **total_beds** | Integer | 总床位数 | 该病房内配置的床位总数。 |
| **occupied_beds** | Integer | 已用床位数 | 当前已经被患者占用的床位数量。 |
| **available_beds** | Integer | 可用床位数 | 当前空闲可供使用的床位数量。通常等于 `total_beds` - `occupied_beds`。 |
| **occupancy_rate** | Float | 占用率 | 床位使用率百分比，计算公式为 `(occupied_beds / total_beds) * 100`。 |
| **timestamp** | DateTime | 时间戳 | 数据记录的具体日期和时间，例如 `2023-12-01`。 |
| **special_status** | String | 特殊状态 | 描述当前床位或病房的状态，例如 `正常`、`满床`、`临时关闭` 等。 |

## 补充说明

- **数据粒度**：数据是按“病房”级别记录的，即每一行代表一个特定时间点上某个病房的床位状态。
- **关联关系**：
  - 一个医院 (`hospital_id`) 包含多个科室 (`department_id`)。
  - 一个科室 (`department_id`) 包含多个病房 (`ward_id`)。
- **计算逻辑**：
  - `available_beds` = `total_beds` - `occupied_beds`
  - `occupancy_rate` 反映了床位的紧张程度，高占用率可能意味着医疗资源紧张。
