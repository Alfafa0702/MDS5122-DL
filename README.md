# CIFAR-10图像分类 - CuPy实现

这个项目是使用CuPy从零实现的深度学习框架，用于CIFAR-10图像分类任务。项目完全使用CuPy重新实现了PyTorch中的核心组件，包括网络层、优化器和训练流程。PyTorch版本可见 `dl-hw1.ipynb` Variant5,包含了数据增强、warmup+cosine学习率调整、AdamW和更深的网络。

## 环境要求

- Python 3.6+
- CuPy
- NumPy
- PyTorch (仅用于数据加载 torch.utils.data.DataLoader)
- torchvision (仅用于数据加载 torchvision.datasets 和数据增强 torchvision.transforms)
- CUDA支持的GPU

## 安装依赖

```bash
pip install numpy cupy==13.2  # 适配cuda121
pip install torch torchvision
```

## 项目结构

- `variant5_cupy.py`: 主要实现文件，包含所有模型组件和训练代码
  - 网络层实现
  - 优化器实现
  - 学习率调整
  - 训练和评估函数
  - 数据处理和增强

## 实现的组件

### 1. 基础组件
- `Module`: 基础模块类
- `Sequential`: 序列容器
- 数据转换工具(to_cpu/to_gpu)

### 2. 网络层
- `Conv2d`: 2D卷积层
- `BatchNorm2d/BatchNorm1d`: 批归一化层
- `ReLU`: 激活函数
- `MaxPool2d`: 最大池化层
- `Dropout`: 随机失活层
- `Linear`: 全连接层
- `Flatten`: 展平层

### 3. 损失函数和优化器
- `CrossEntropyLoss`: 交叉熵损失
- `AdamW`: 带权重衰减的Adam优化器

### 4. 学习率调整 `LRScheduler`
- Warmup阶段：学习率从initial_lr线性增加到max_lr
- Constant阶段：学习率保持max_lr
- Cosine退火阶段：学习率从max_lr按余弦函数衰减

### 4. 数据增强
- 随机裁剪
- 随机水平翻转
- 随机旋转
- 随机仿射变换
- 颜色抖动
- 随机擦除

## 模型架构

Variant4模型架构如下：
1. 卷积块：
   - 7个卷积层
   - BatchNorm层
   - ReLU激活
   - MaxPool层
2. 全连接块：
   - 4个全连接层
   - Dropout层
   - BatchNorm层
   - ReLU激活

## 使用方法

1. 克隆仓库：
```bash
git clone [repository-url]
cd [repository-name]
```

2. 运行训练：
```bash
python variant4_cupy.py
```

## 训练参数

- 批次大小：64
- 学习率：3e-4
- 权重衰减：1e-6
- 训练轮数：64
- 优化器：AdamW
- Dropout率：0.3

## 性能

模型在CIFAR-10测试集上的表现：
- 参数量：约127.06M
- 训练时间：视GPU性能而定
- 预期准确率：~93%

## 注意事项

1. 确保有足够的GPU内存
2. 建议使用支持CUDA的GPU进行训练
3. 可以通过修改`build_variant5_model()`函数来调整模型架构
4. 可以通过修改`train()`函数中的参数来调整训练过程

## 参考

- [CIFAR-10数据集](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CuPy文档](https://docs.cupy.dev/en/stable/) 