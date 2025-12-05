本项目针对 Home Credit Default Risk 竞赛（金融信贷违约预测），构建了一套Deep Learning (NN) 与 LightGBM深度融合的解决方案。
针对表格数据高维稀疏、噪声大及正负样本极度不平衡的痛点，本项目并未止步于传统的特征工程 + XGBoost/LightGBM 方案，而是探索了深度学习在表格数据上的 SOTA 实践。
核心创新点在于引入了 DAE (Denoising AutoEncoder) 进行自监督预训练，并结合 ResNet 残差网络架构,此时会发现结果auc在测试集上并不突出,但是在接下来使用了深度的特征融合：
>>1.将此时DNN训练出来的潜在特征(选取训练标签y_train有较强相关性)与x_train,x_test重新融合x_train_aug_df,x_test_aug_df,重新喂入train_lgb_oof()函数中去,AUC可以达到0.787.
>>2.进一步的利用这一次较好的对验证集的预测结果构建伪标签(选取正样本预测概率在0.7以上,负样本的预测概率在0.05以下),这里的5折交叉验证不是直接利用上述的train_lgb_oof(),是因为避免信息泄露只对每一次fold中的训练集添加拼接伪标签数据.效果很显著的我们可以看到AUC达到0.789
>>3.由于KNN在局部特征增强的优势,进一步的提取到knn_train,knn_test,并于x_train_aug_df,x_test_aug_df进一步的融合,此时再次喂入train_lgb_oof(),测试得新的AUC

>下一步想做的是进行Rank Averaging
>在特征工程中涉及到的类别特征也是很多的,此时使用CatBoost对类别特征的交互有比较强的优势,进行train_catboost_oof(),利用rankdata将对上一步得操作下最优的lgb_test_preds进行构建,并于cat_test_preds分配权重会得到新的AUC

>>想要在这些基础上得到优秀的UC结果,必须要有详细的特征工程和特征分析的环节.具体可见One-Hot.ipynb
