# Kaggle Project Template

这是 Kaggle 比赛项目模板。

## 使用方式

1. 复制 `_template` 文件夹。
2. 将复制后的文件夹重命名为具体比赛名。
3. 为每个比赛建立独立 Git 仓库。

## 目录约定

- 原始数据放在 `data/raw`
- 处理后数据放在 `data/processed`
- 算法文件放在 `src/algorithms`
- 特征工程放在 `src/features`
- 提交文件放在 `outputs/submissions`
- 模型文件放在 `models`

## 说明

这是一个 Kaggle 项目总入口。
后续采用“一个比赛一个文件夹，一个比赛一个独立 Git 仓库”的方式管理。
新比赛时直接复制 `_template` 再重命名。
