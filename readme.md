# 基于机器学习的安卓恶意应用检测系统

------



[TOC]

------

### 项目简介

本项目将APK（Android应用文件）合集使用apktool反编译后获取Smail文件合集，从中提取出每个APK对应的Dalvik字节码，再将字节码简化为指令集符号后进行N-Gram编码从而提取特征。随后使用了随即森林、GBDT、决策树等8种传统机器学习算法和多层感知机、双向LSTM这两种深度学习算法来训练模型和测试模型，对比选择出最后的模型，最终效果最好的是多层感知机，精确率达到97.8%。

------

### 参与人员
张原钰
李珠源
赵贞宇


