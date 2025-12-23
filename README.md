# 🏦 Home Credit Default Risk: 基于 DAE-ResNet 与 LightGBM 的深度融合预测模型

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-green)
![Status](https://img.shields.io/badge/Status-Optimization-orange)

> **核心亮点**：本项目针对金融风控场景中表格数据高维稀疏、噪声大及正负样本极度不平衡的痛点，构建了一套结合 **自监督学习 (DAE)**、**残差网络 (ResNet)** 与 **梯度提升树 (LightGBM)** 的混合架构。通过**伪标签 (Pseudo-labeling)** 半监督学习策略与 **KNN 局部拓扑特征**融合，在测试集上取得了显著的性能提升。

## 📖 项目背景 (Background)
在金融信贷违约预测任务中，传统的 GBDT 方案往往面临特征挖掘的瓶颈，而单纯的深度学习模型在结构化数据上表现不佳。本项目旨在探索 **Deep Learning (NN)** 与 **Tree-based Models** 的最佳结合点，通过特征表示学习（Representation Learning）挖掘潜在风险因子。

## 🛠️ 核心架构与创新点 (Architecture & Innovations)

### 1. 深度特征提取 (Deep Representation Learning)
- **DAE (Denoising AutoEncoder) 自监督预训练**：
  - 针对含噪表格数据，使用 **Swap Noise** 进行数据增强，训练 DAE 进行无监督特征重构，有效提取了数据的鲁棒潜在分布。
- **ResNet 残差网络架构**：
  - 摒弃传统的 MLP，搭建基于 ResNet 的分类器，利用 Skip Connection 缓解深层网络的梯度消失问题，捕捉非线性交互特征。
- **特征筛选与正交性检查**：
  - 提取 DNN 中间层输出 (Embedding)，通过 Pearson 相关性分析，筛选出与原始特征互补的 **Top 潜在特征** (Latent Features)。

### 2. 策略融合与半监督学习 (Advanced Strategy)
- **特征层融合 (Feature-Level Fusion)**：
  - 将 DNN 提取的潜在特征、**KNN (K-Nearest Neighbors)** 提取的局部近邻统计特征与原始特征进行拼接。
- **鲁棒的伪标签策略 (Robust Pseudo-labeling)**：
  - 利用测试集的分布信息，通过 **多折交叉验证 (K-Fold CV)** 预测测试集。
  - **创新点**：针对验证集泄漏问题，严格执行 Fold 内部融合；针对样本不平衡，实施 **不同置信度阈值筛选 (正样本>0.70, 负样本<0.02)**，构建高质量伪标签数据集，显著提升模型泛化能力。

## 📊 实验结果 (Performance)

通过逐步叠加不同策略，模型在验证集与测试集上的 AUC 指标稳步提升：

| 实验阶段 (Stage) | 方法描述 (Methodology) | AUC Score | 提升 (Improvement) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 原始特征 + LightGBM (5-Fold) | 0.7869 | - |
| **Exp 1** | + ResNet 潜在特征融合 (Latent Features) | 0.7870 | 🔺 Slight |
| **Exp 2** | + **Pseudo-labeling (Semi-supervised)** | **0.7893** | **Significant** |
| **Exp 3** | + KNN Features + Feature Selection | 0.7886 | 🔄 Optimization |
| **Exp 4** | Exp 2 + CatBoost (5-Fold) 进行逻辑回归 | **0.7921** | 🚀Super Model| 

> *注 1：伪标签策略使得 AUC 突破了 0.789 的瓶颈，证明了利用未标记数据的有效性。*
> 
> *注 2 ：使用KNN并未让结果再一次的提高, 说明此时提取的KNN特征和Exp 2中的特征有很大的相关性,且增大了模型的复杂度。*
> 
> *注 3： 使用CatBoost来增强类别特征的交互性,但是训练速度很慢,需要耐心的去调整参数.*

## 📂 项目结构 (Structure)

```text
├── data_EDA.ipynb        # 数据探索性分析 (分布检查、缺失值处理)
├── One_Hot.ipynb         # 特征工程 (One-Hot, Target Encoding, 聚合统计)
├── DAE.ipynb             # 深度学习模块 (DAE预训练 + ResNet微调 + 特征提取)
├── models/               # 模型训练脚本
│   ├── train_lgb.py      # LightGBM 训练与伪标签逻辑实现
│   └── train_cat.py      # CatBoost 训练 (Plan)
├── README.md             # 项目说明文档
