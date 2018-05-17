# 基于hourglass的人体关键点检测

## 目标

对图像中人体的16个关键点进行检测，使用MPII数据集

## 方法

本项目只验证关键点检测方法，不估计人体姿态，所以虽然数据中有多人场景，依然选择hourglass作为网络模型

使用**2D高斯函数**来构建学习目标（heatmap）。将某一关键点的ground-truth作为中心点，这样一来，中心点处将具有最高的得分，越远离中心点，得分将越低。公式地表示，则有

  ![学习目标构建公式](images/f1.png)

### 网络结构
  
输入为64*64RGB图像，hourglass结构选择4次降采样，channel数512,共2个hourglass结构级联。输出(n,16,64,64)即16个关键点。

hourglass 模型图

  ![hourglass模型](images/hour.png)

## 运行

* training
```bash
python3 train.py
```

