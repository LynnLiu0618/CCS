# CCS-FD

[English](README.md) | 简体中文

**CCS-FD 是一种面向开放集复合故障诊断的方法。** 它利用单故障测量数据学习各故障成分的特征，综合不同部位的证据，预测哪些故障可能同时发生。预测组合决定需要核验的部件状态，算法再根据该状态下的邻近参考样本，判断实测响应是否支持这一诊断。

本仓库提供论文 *Open-Set Compound Fault Diagnosis via Composition Prediction and Local Support Learning* 在 BJTU-RAO 转向架数据集上的实现。论文以轴箱轴承为例说明候选状态的局部核验，并提供五次独立训练得到的模型。

## 快速开始

在仓库根目录运行以下命令。需要 Python 3.10 或更高版本，推理依赖 NumPy 和 PyTorch。论文实验环境使用 Python 3.10、NumPy 1.26.4 和 PyTorch 2.7.1。第二条命令使用生成的输入在 CPU 上测试，可以在下载数据前运行。

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

将同步采集的信号窗口保存为形状为 **(N, 2048, 18)** 的数值数组，其中 N 是窗口数，后两维依次为采样点和通道。下面的命令读取 `windows.npy`，将每个窗口的诊断结果写入 CSV。

```sh
python -m ccs_fd windows.npy --device cpu --output predictions.csv
```

| CSV 字段 | 含义 |
| --- | --- |
| `window` | 输入窗口的序号，从 0 开始 |
| `candidate` | 成分预测给出的候选故障标签 |
| `squared_distance` | 实测响应在特征空间中到候选状态下第五近邻参考样本的平方距离 |
| `prediction` | 最终诊断。平方距离小于或等于该状态的标定阈值时接受候选标签，超过阈值时返回 `UNKNOWN` |

若已有本仓库特征提取器生成的 `(N, 2628)` 数组，可用 `--features` 直接输入特征。默认模型的随机种子为 `1994217509`，可用 `--checkpoint ccs_fd/checkpoints/1454668731.pt` 切换到另一次实验的模型。

也可以在 Python 中分别读取候选标签、最终诊断和核验距离。

```python
import numpy as np
from ccs_fd import Predictor

model = Predictor(device="cpu")
windows = np.load("windows.npy", allow_pickle=False)
result = model.predict_windows(windows, return_detail=True)
print(result["label"][:5])     # 核验前的候选标签
print(result["pred"][:5])      # 核验后的最终诊断
print(result["distance"][:5])  # 到第五近邻参考样本的平方距离
```

## BJTU-RAO 数据与故障标签

随仓库提供的模型使用 64 kHz 同步采样信号，每个窗口长 32 ms。保留原始信号单位，按 **CH1–CH9、CH11–CH19** 的顺序排列 18 个通道，跳过转速通道 CH10。这些通道包括 6 路电机加速度、3 路电机电流、6 路齿轮箱加速度和 3 路左轴箱加速度。每个窗口提取 900 维电机特征、600 维齿轮箱特征和 1128 维轴承特征。

| 标签 | 实测故障 |
| --- | --- |
| A | 电机短路 |
| B | 齿轮缺齿 |
| C / D / E | 分别为左轴箱轴承内圈、外圈、保持架故障 |
| AB | 电机短路与齿轮缺齿同时发生 |
| CD | 轴承内圈和外圈同时受损 |
| F | 轴承滚动体故障，仅用于评估未知故障拒识，正确输出为 `UNKNOWN` |

成分预测综合电机、齿轮箱和轴承的证据，局部核验则仅使用轴箱响应。候选 A、B、AB 均选择参考状态 H，表示轴承未设置故障；候选 C、D、E、CD 分别选择对应的轴承状态。

原始记录可从数据集作者提供的目录获取。下载后，将下文命令中的 `/path/to/BJTU_RAO_Bogie_Datasets` 替换为本机存放各故障文件夹的目录。[BJTU-RAO 数据集](https://drive.google.com/drive/folders/1RlZvFw-v07VvsL2Ni9cS7iFrTPDIhn2r?usp=sharing) · [试验台与数据集论文，Ding 等，2024](https://doi.org/10.1016/j.ymssp.2024.111175)

## 复现论文结果

以下命令安装原始数据读取依赖，从实测记录生成评估特征，并使用五个模型逐一评估，结果保存到 `report.json`。

```sh
python -m pip install -e ".[prepare]"
python -m training.prepare_data evaluation --data-root /path/to/BJTU_RAO_Bogie_Datasets --output-dir data
python evaluate.py data/evaluation.npz --output report.json
```

评估包含 8 种实测故障状态和 9 种工况，每种状态在每种工况下取 62 个窗口，共 4464 个。五次实验使用同一试验台、相同的时间划分。下表报告窗口级指标在五次实验中的均值及样本标准差，单位为 %。

| 指标 | 均值 ± 样本标准差 |
| --- | ---: |
| 总体准确率（完整标签匹配） | 94.45 ± 0.36 |
| 单故障准确率（完整标签匹配） | 95.86 ± 0.46 |
| 电机与齿轮复合故障 AB 召回率 | 93.91 ± 1.27 |
| 轴承内外圈复合故障 CD 召回率 | 92.37 ± 1.58 |
| 未知滚动体故障 F 拒识率 | 90.07 ± 1.11 |
| 已知故障窗口误拒率 | 1.47 ± 0.21 |
| AUROC | 99.29 ± 0.07 |

每个模型文件对应一次实验，包含成分预测器、局部核验编码器、归一化参数、参考样本的特征表示和标定阈值。[逐次实验结果](results/paper_runs.csv) · [完整精度汇总](results/paper_summary.csv) · [模型 SHA-256 校验值](results/checkpoints.json) · [与论文结果的核对说明（英文）](results/README.md)

## 使用源数据重新训练

安装上文的 `.[prepare]` 依赖后，运行以下命令准备源数据，并从头训练指定随机种子对应的模型。

```sh
python -m training.prepare_data source --data-root /path/to/BJTU_RAO_Bogie_Datasets --output-dir data
python -m training.fit --data data --seed 1994217509 --output outputs/model.pt
```

训练使用实测单故障 A–E，以及由源域单故障窗口合成的 AB 和 CD。实测 AB、CD 和 F 仅用于评估。各阶段按每条原始记录的时间划分，区间均为左闭右开。

| 阶段 | 时间区间（秒） |
| --- | --- |
| 训练 | [0, 3.488) |
| 标定 | [5.512, 6.504) |
| 验证 | [6.504, 7.496) |
| 评估 | [8.012, 9.996) |

验证窗口用于核对候选故障标签，并检查各状态参考库中的距离是否按近邻序号从小到大排列。源数据文件的字段、五个随机种子和模型导出方法见[训练说明（英文）](training/README.md)。
