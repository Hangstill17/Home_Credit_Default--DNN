# 🚀 Tabular Transformer 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **"Beating Gradient Boosting Decision Trees on Tabular Data with Deep Learning."**

## 📖 Introduction (项目背景简介)

在金融风控场景中，表格数据通常面临高维稀疏、特征异质性强、非线性交互复杂等挑战。虽然 LightGBM/XGBoost 是该领域的主流模型，在Kaggle比赛上选手们大都会因构造复杂的特征工程来获取较高的AUC，但对于特征发挥的作用我们的确可以使用累计增益率来可视化哪一特征比较重要，但同时我们似乎也会忽略特征之间的交互作用。目前transformer的流行也使得我们有了进一步的方向。

本项目设计了一套**端到端的深度学习预测系统**。通过引入 **Transformer Encoder** 捕捉全局上下文，结合 **SENet** 进行动态特征加权，并利用 **DCN-v2** 显式建模高阶交互，最终在真实金融数据集上实现了 **AUC 0.7868**，超越了经过精细调优的 LightGBM 模型。

## ✨ Key Features (核心创新)

### 1. Dual-Stream Hybrid Architecture (双流混合架构)
针对数值与类别特征的异质性，采用分治策略：
*   **类别特征**: 利用 **Transformer Encoder** (Self-Attention) 捕捉全局特征交互。
*   **数值特征**: 引入 **SENet (Squeeze-and-Excitation)** 模块，实现端到端的**软特征选择 (Soft Feature Selection)**，有效抑制了长尾分布中的噪声干扰。

### 2. Advanced Feature Engineering (前沿特征工程)
*   **Periodic Embeddings**: 引入基于傅里叶特征的**周期性嵌入层**替代传统的 Linear 映射，显著提升了模型对数值特征（如收入、年龄）中**非单调、高频模式**的拟合能力，同时也大大加快了模型训练的速度

### 3. Explicit Feature Interaction (显式特征交叉)
*   **DCN-v2 (Deep & Cross Network)**: 在网络末端引入低秩分解 (Low-Rank) 的 Cross Network，弥补了双流架构中特征割裂的问题，显式捕获了跨域（数值 $\times$ 类别）的高阶组合特征。

### 4. Robust Data Processing (鲁棒工程化)
*   **Robust Label Encoder**: 自研编码器，支持 `min_obs` 阈值过滤，并采用鲁棒的 Unknown 映射策略，有效处理测试集未见类别 (Unseen Categories)。
*   **DAE Pre-training**: (Optional) 集成了基于 **Swap Noise** 的去噪自编码器预训练流程，探索了无监督学习在特征表征上的潜力。

## 📊 Performance (实验结果)

在包含 30万+ 样本的金融信贷数据集上，各模型表现如下：

| 模型 | AUC  | 提升 | 备注 |
| :--- | :--- | :--- | :--- |
| **LightGBM** | 0.7845 | - | 全局最优参数下的AUC |
| MLP + ResNet | 0.7650 | -1.95% | 简单的使用FFN |
| **DeepTabular** | **0.7868** | **+0.23%** | **transformer捕捉高阶交互** |

> *注：在金融风控场景下，0.2% 的 AUC 提升通常意味着显著的坏账率降低和巨大的业务收益。*

## 🛠️ 注意事项：

整个模型的训练能跑过LGB虽然是一大进步，但是更为关键的是如何去设法突破到**0.79**。

也许我之前的过程不太完美，但这是我第一次做出成果并取得愉悦感的时候，写出来也是取悦自己，也是作为经验分享。

LGB模型的贝叶斯调优是在**DAE.ipynb**文件下。