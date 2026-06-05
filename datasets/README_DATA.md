# Datasets / 数据集

The processed CMU-MOSI and CMU-MOSEI feature pickles are **not** stored in this
repository (too large for GitHub). Download them from the link below and place
them exactly as shown.

处理好的 CMU-MOSI / CMU-MOSEI 特征 pickle **不在**本仓库中（GitHub 放不下）。
请从下面的链接下载，并严格按下述目录结构放置。

## Download / 下载

通过网盘分享的文件：mosi+mosei_all.zip
链接: https://pan.baidu.com/s/1CS8L8W0irjqO2v0HnoLqsg 提取码: 0524 --来自百度网盘超级会员v6的分享

Extract the archive so that `MOSI/` and `MOSEI/` sit directly under `datasets/`.
The raw `*_noalign.pkl` source files (used only to rebuild features) are **not**
required to train/evaluate and can be deleted to save space.

解压后让 `MOSI/`、`MOSEI/` 直接位于 `datasets/` 下。原始 `*_noalign.pkl`（仅用于重建特征）
**不需要**，可删除以节省空间。

## Expected layout / 目录结构

```
datasets/
├── MOSI/
│   ├── train.pkl                       # standard split / 标准划分
│   ├── dev.pkl
│   ├── test.pkl
│   ├── split_dataset_2/                # for --iid_setting / --ood_setting (2-class dir)
│   │   ├── train.pkl  ├── dev.pkl
│   │   ├── test_IID.pkl  └── test_OOD.pkl
│   └── split_dataset_7classes_1/       # for --iid_setting/--ood_setting + --seven_class
│       ├── train.pkl  ├── dev.pkl
│       ├── test_IID.pkl  └── test_OOD.pkl
└── MOSEI/
    ├── train.pkl                       # standard split / 标准划分
    ├── dev.pkl
    ├── test.pkl
    ├── split_dataset_2/
    │   ├── train.pkl  ├── dev.pkl
    │   ├── test_IID.pkl  └── test_OOD.pkl
    └── split_dataset_7classes_1/
        ├── train.pkl  ├── dev.pkl
        ├── test_IID.pkl  └── test_OOD.pkl
```

Which files each run needs / 每种运行需要的文件：

| Command / 命令 | Files used / 用到的文件 |
|---|---|
| `python main.py` *(standard)* | `<DS>/{train,dev,test}.pkl` |
| `--iid_setting` | `<DS>/split_dataset_2/{train,dev,test_IID}.pkl` |
| `--ood_setting` | `<DS>/split_dataset_2/{train,dev,test_OOD}.pkl` |
| `--iid_setting --seven_class` | `<DS>/split_dataset_7classes_1/{train,dev,test_IID}.pkl` |
| `--ood_setting --seven_class` | `<DS>/split_dataset_7classes_1/{train,dev,test_OOD}.pkl` |

The code resolves this folder as `<repo_root>/datasets/<MOSI|MOSEI>/`
(see `CPAP-my/config.py`), so keep the `datasets/` folder a sibling of `CPAP-my/`.

代码按 `<仓库根>/datasets/<MOSI|MOSEI>/` 解析（见 `CPAP-my/config.py`），
请保持 `datasets/` 与 `CPAP-my/` 同级。
