# CPAP — A Multimodal Sentiment Analysis Model for Causal Perception Enhancement and Ambiguity Perception Fusion

*因果感知增强与歧义感知融合的多模态情感分析模型*

*[English](#english) · [中文](#中文)*

> Minimal, reproducible release of the **CPAP** model for multimodal sentiment
> analysis on CMU-MOSI and CMU-MOSEI. This repository contains only the code path
> needed to train and evaluate a single model run.
>
> **CPAP** 在 CMU-MOSI / CMU-MOSEI 上做多模态情感分析的**最小可复现**代码，仅包含
> 单次训练 + 评估所需的链路。

---

## English

### Overview

CPAP is a Transformer-based multimodal sentiment regression model. The codebase
implements, end to end:

- **Per-modality encoders** — text (BERT / RoBERTa), acoustic & visual RNNs.
- **Debiasing self-attention** over each unimodal stream.
- **A KMeans confounder dictionary** (`npy_folder/`, precomputed and shipped) used
  by the causal perception enhancement modules.
- **APF** — an adaptive probabilistic fusion module with KL-based gating
  (ambiguity perception fusion).

The final task is sentiment-score regression, reported with MAE, correlation,
7-class accuracy, and binary Acc/F1.

### Repository layout

```
MSA-CPAP/
├── CPAP-my/                # all source code (run commands from inside this folder)
│   ├── main.py             # entry point: load data → train → evaluate → save CSV
│   ├── config.py           # all command-line arguments & paths
│   ├── data_loader.py      # batching + HuggingFace tokenization
│   ├── create_dataset.py   # loads the processed train/dev/test pickles
│   ├── solver.py           # training / evaluation loop
│   ├── model.py            # full CPAP model
│   ├── modules/            # encoders, attention, debiasing, APF, ...
│   ├── utils/              # metrics, EMA, KMeans helper, checkpoint I/O
│   └── npy_folder/         # precomputed KMeans centers (~5 MB, included)
├── datasets/               # download separately — see datasets/README_DATA.md
├── requirements.txt
└── README.md
```

### Requirements

- **Python 3.12**, a **CUDA-capable GPU** (the model uses `.cuda()` directly — CPU is not supported).
- Install PyTorch matching your CUDA, then the rest:

```bash
# 1) PyTorch (example: CUDA 12.1)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# 2) remaining dependencies
pip install -r requirements.txt
```

The text encoders (`bert-base-uncased`, `cardiffnlp/twitter-roberta-base-sentiment`)
are downloaded automatically from the HuggingFace Hub on first run and cached
locally — no manual weight download needed.

### Data

Download the processed MOSI/MOSEI pickles and place them under `datasets/` as
described in [datasets/README_DATA.md](datasets/README_DATA.md). Minimal bundle:
`datasets/{MOSI,MOSEI}/{train,dev,test}.pkl`.

### Run

Always run from inside `CPAP-my/` (paths like `npy_folder/` are resolved relative
to the working directory):

```bash
cd CPAP-my

# train + evaluate on MOSI (defaults are tuned for MOSI)
python main.py --dataset mosi

# train + evaluate on MOSEI
python main.py --dataset mosei
```

**Robustness (IID / OOD) splits.** By default the standard split is used. To
train/evaluate on the IID or OOD test set instead:

```bash
python main.py --dataset mosi --iid_setting                 # IID test (split_dataset_2)
python main.py --dataset mosi --ood_setting                 # OOD test (split_dataset_2)
python main.py --dataset mosi --ood_setting --seven_class   # OOD test, 7-class split dir
```

`--iid_setting` and `--ood_setting` are mutually exclusive. These require the
`split_dataset_*` folders from the download — see [datasets/README_DATA.md](datasets/README_DATA.md).

Results are written to `CPAP-my/pre_trained_best_models_<mosi|mosei>/…_result.csv`,
and the best checkpoint to the same folder.

### Key arguments

Defaults live in `config.py`; many help strings note the recommended MOSEI value
(e.g. *"mosi for 3, mosei for 5"*). The most useful flags:

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `mosi` | `mosi` or `mosei` |
| `--text_encoder` | `roberta` | `roberta` or `bert` text backbone |
| `--batch_size` | `64` | batch size |
| `--num_epochs` | `40` | training epochs |
| `--patience` | `5` | early-stop patience |
| `--lr_main` / `--lr_bert` | tuned | main / text-encoder learning rates |
| `--beta` | `0.1` | weight of the MMILB likelihood term (MOSEI: `0.25`) |
| `--use_kmean` | on | use shipped KMeans centers (pass the flag to disable) |
| `--iid_setting` / `--ood_setting` | off | evaluate on the IID / OOD robustness split (mutually exclusive) |
| `--seven_class` | off | use the 7-class `split_dataset_7classes_1` directory for IID/OOD |
| `--seed` | `1111` | random seed |
| `--device` | `cuda` | compute device |

See `python main.py -h` for the full list (debias layers, APF, cross-attention dims, etc.).

### Notes & scope

- This is a **minimal** release covering a single training/evaluation run. Both the
  standard split and the IID/OOD robustness splits are supported (see flags above).
- Tooling not needed to reproduce a single run (hyperparameter search, plotting
  utilities, backups) has been removed.

### Citation

If you use this code, please cite:

> Chen Kejia, Zhao Xiaofeng, Zhou Xiukao. *A Multimodal Sentiment Analysis Model
> for Causal Perception Enhancement and Ambiguity Perception Fusion.* Data Analysis
> and Knowledge Discovery, 1–22. DOI: 10.11925/infotech.2096-3467.2025.0392

```bibtex
@article{chen2025cpap,
  title   = {A Multimodal Sentiment Analysis Model for Causal Perception Enhancement and Ambiguity Perception Fusion},
  author  = {Chen, Kejia and Zhao, Xiaofeng and Zhou, Xiukao},
  journal = {Data Analysis and Knowledge Discovery},
  pages   = {1--22},
  year    = {2025},
  doi     = {10.11925/infotech.2096-3467.2025.0392}
}
```

License: released under the [MIT License](LICENSE).

---

## 中文

### 简介

CPAP 是一个基于 Transformer 的多模态情感回归模型。本代码端到端实现了：

- **单模态编码器** —— 文本（BERT / RoBERTa）、语音与视觉 RNN。
- 对各单模态流的 **去偏自注意力（debias self-attention）**。
- **KMeans 混杂字典**（`npy_folder/`，已预计算并随仓库提供），供因果感知增强模块使用。
- **APF** —— 基于 KL 门控的自适应概率融合模块（歧义感知融合）。

最终任务是情感分数回归，报告 MAE、相关系数、7 分类准确率以及二分类 Acc/F1。

### 目录结构

```
MSA-CPAP/
├── CPAP-my/                # 全部源码（运行命令需在此目录内执行）
│   ├── main.py             # 入口：加载数据 → 训练 → 评估 → 保存 CSV
│   ├── config.py           # 所有命令行参数与路径
│   ├── data_loader.py      # 组 batch + HuggingFace 分词
│   ├── create_dataset.py   # 加载处理好的 train/dev/test pickle
│   ├── solver.py           # 训练 / 评估循环
│   ├── model.py            # 完整 CPAP 模型
│   ├── modules/            # 编码器、注意力、去偏、APF 等
│   ├── utils/              # 指标、EMA、KMeans 工具、模型存取
│   └── npy_folder/         # 预计算的 KMeans 质心（约 5 MB，已包含）
├── datasets/               # 需单独下载，见 datasets/README_DATA.md
├── requirements.txt
└── README.md
```

### 环境要求

- **Python 3.12**，**支持 CUDA 的 GPU**（模型中直接使用 `.cuda()`，不支持 CPU 运行）。
- 先安装与你 CUDA 匹配的 PyTorch，再装其余依赖：

```bash
# 1) PyTorch（示例：CUDA 12.1）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# 2) 其余依赖
pip install -r requirements.txt
```

文本编码器（`bert-base-uncased`、`cardiffnlp/twitter-roberta-base-sentiment`）会在
首次运行时从 HuggingFace Hub 自动下载并缓存到本地，**无需手动下载权重**。

### 数据

按 [datasets/README_DATA.md](datasets/README_DATA.md) 下载处理好的 MOSI/MOSEI pickle，
放入 `datasets/`。最小数据包：`datasets/{MOSI,MOSEI}/{train,dev,test}.pkl`。

### 运行

务必在 `CPAP-my/` 目录内运行（`npy_folder/` 等路径相对工作目录解析）：

```bash
cd CPAP-my

# 在 MOSI 上训练 + 评估（默认超参为 MOSI 调好的值）
python main.py --dataset mosi

# 在 MOSEI 上训练 + 评估
python main.py --dataset mosei
```

**鲁棒性（IID / OOD）划分。** 默认用标准划分。要改用 IID 或 OOD 测试集：

```bash
python main.py --dataset mosi --iid_setting                 # IID 测试（split_dataset_2）
python main.py --dataset mosi --ood_setting                 # OOD 测试（split_dataset_2）
python main.py --dataset mosi --ood_setting --seven_class   # OOD 测试，7 分类划分目录
```

`--iid_setting` 与 `--ood_setting` 互斥。这些需要下载包里的 `split_dataset_*`
子目录——见 [datasets/README_DATA.md](datasets/README_DATA.md)。

结果写入 `CPAP-my/pre_trained_best_models_<mosi|mosei>/…_result.csv`，
最优 checkpoint 也保存在同一目录。

### 主要参数

默认值见 `config.py`；许多帮助文本里标注了 MOSEI 的推荐值（如 *"mosi for 3, mosei for 5"*）。
最常用的开关：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--dataset` | `mosi` | `mosi` 或 `mosei` |
| `--text_encoder` | `roberta` | 文本主干 `roberta` / `bert` |
| `--batch_size` | `64` | 批大小 |
| `--num_epochs` | `40` | 训练轮数 |
| `--patience` | `5` | 早停耐心值 |
| `--lr_main` / `--lr_bert` | 已调 | 主干 / 文本编码器学习率 |
| `--beta` | `0.1` | MMILB 似然项权重（MOSEI：`0.25`） |
| `--use_kmean` | 开启 | 使用随仓库提供的 KMeans 质心（带上该 flag 则关闭） |
| `--iid_setting` / `--ood_setting` | 关闭 | 在 IID / OOD 鲁棒性划分上评估（二者互斥） |
| `--seven_class` | 关闭 | IID/OOD 时使用 7 分类目录 `split_dataset_7classes_1` |
| `--seed` | `1111` | 随机种子 |
| `--device` | `cuda` | 计算设备 |

完整参数（去偏层数、APF、跨注意力维度等）见 `python main.py -h`。

### 说明与范围

- 本仓库是**最小**释放版，覆盖单次训练 + 评估。标准划分与 IID/OOD 鲁棒性划分均支持
  （见上文开关）。
- 复现单次运行不需要的工具（超参搜索、绘图脚本、备份）已移除。

### 引用

如果本代码对你有帮助，请引用：

> 陈可嘉, 赵晓锋, 周修考. 因果感知增强与歧义感知融合的多模态情感分析模型[J].
> 数据分析与知识发现, 1-22. DOI：10.11925/infotech.2096-3467.2025.0392

```bibtex
@article{chen2025cpap,
  title   = {A Multimodal Sentiment Analysis Model for Causal Perception Enhancement and Ambiguity Perception Fusion},
  author  = {Chen, Kejia and Zhao, Xiaofeng and Zhou, Xiukao},
  journal = {Data Analysis and Knowledge Discovery},
  pages   = {1--22},
  year    = {2025},
  doi     = {10.11925/infotech.2096-3467.2025.0392}
}
```

许可证：基于 [MIT License](LICENSE) 开源。
