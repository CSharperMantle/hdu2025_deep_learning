#import "@preview/ouset:0.2.0": underset
#import "@preview/booktabs:0.0.4": *

#import "../template/hdu-report-typst/template/template.typ": *

#show: booktabs-default-table-style
#show: project.with(
  title: "《深度学习》作业2",
  subtitle: [基于MLP的MNIST与FashionMNIST图像分类],
  class: "计算机科学英才班",
  department: "卓越学院",
  authors: "鲍溶",
  author_id: "23060827",
  date: (2025, 10, 21),
  cover_style: "hdu_report",
)

#show link: underline

#toc()
#pagebreak()

= 方法简介

== 非线性函数

深度神经网络中常用非线性函数实现非平凡的函数拟合。常见的非线性激活函数包括：

- *整流线性单元（ReLU）*：$f_"ReLU" (x) = max{0, x}$，在正值区域保持线性，负值区域置零。
- *Sigmoid函数*：$f_"sigmoid" (x) = 1/(1 + e^(-x))$，将输入映射到$(0, 1)$区间，常用于二分类输出层。
- *Tanh函数*：$f_"tanh" (x) = tanh(x)$，将输入映射到$(-1, 1)$区间，输出以零为中心。
- *Softmax函数*：$f_"softmax" (bold(x))_i = e^(x_i) / (sum_j e^(x_j))$，用于多分类问题的输出层，将向量归一化为概率分布。

== 感知机模型

单层感知机的计算过程可表示为：

$ p(bold(x), bold(A), bold(b)):  bold(y) = f_"act" (bold(x) bold(A)^top + bold(b)) $

其中$f_"act"$为激活函数，$bold(A)$与$bold(b)$为可训练参数，分别对应权重矩阵与偏置向量。

== 深度神经网络

将每层的计算表示为函数$f_"layer" (bold(x), theta)$，则深度神经网络可表示为多个层函数的复合：

$ N(bold(x), bold(theta)): bold(y) = (underset(circle, i in I) f_("layer" i)(dot, theta))(bold(x)) $

其中$circle.small$表示函数复合运算，$I$为层索引集合，$bold(theta)$包含所有层的参数。

== 监督学习

给定观测样本$bold(x)_s$及其对应的真实标签$bold(y)_s$，监督学习的训练过程可表述为以下优化问题：

$ hat(bold(theta)) = underset(arg min, bold(theta)) #h(0.3em) f_"loss" (N(bold(x)_s, bold(theta)), bold(y)_s) $

其中$f_"loss"$为定义在预测值空间上的某种距离度量（损失函数）。常用的损失函数包括交叉熵损失、均方误差等。训练过程通过梯度下降等优化算法迭代更新参数$bold(theta)$，使损失函数最小化。

= 实验设置

== 模型结构

如@code:baseline-model 所示，实验中采用PyTorch @pytorch2024 实现MLP模型，包含一个输入层、一个隐藏层、一个输出层，均为简单矩阵相乘（线性）层。前两层使用ReLU激活函数，最后一层使用Softmax函数获取预测结果。

#code(
  ```python
  class MLPBaseline(nn.Module):
      def __init__(self):
          super(MLPBaseline, self).__init__()
          self.fc1 = nn.Linear(784, 128)
          self.fc2 = nn.Linear(128, 64)
          self.fc3 = nn.Linear(64, 10)

      def forward(self, x):
          x = self.fc1(x)
          x = F.relu(x)
          x = self.fc2(x)
          x = F.relu(x)
          x = self.fc3(x)
          x = F.softmax(x, dim=1)
          return x
  ```,
  caption: [基线MLP模型结构。]
) <code:baseline-model>

实验中通过更改前两层的激活函数、增减隐藏层数进行差分比对，测试不同参数组合对模型预测性能的影响。同时，还有一个如@code:conv-model 的卷积模型作为参照，比较不同设计思想的模型在同样数据集上的性能表现。

#code(
  ```python
  class ConvClassifier(nn.Module):
      def __init__(self, p_dropout=0.2):
          super(ConvClassifier, self).__init__()
          # Input: (B, 784) -> (B, 1, 28, 28)
          self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # -> (B, 32, 24, 24)
          self.conv2 = nn.Conv2d(32, 16, kernel_size=5)  # -> (B, 16, 20, 20)
          self.fc1 = nn.Linear(16 * 20 * 20, 128)  # -> (B, 128)
          self.fc2 = nn.Linear(128, 10)  # -> (B, 10)

      def forward(self, x):
          x = x.view(-1, 1, 28, 28)
          x = F.relu(self.conv1(x))
          x = F.relu(self.conv2(x))
          x = x.view(-1, 16 * 20 * 20)
          x = F.relu(self.fc1(x))
          x = F.softmax(self.fc2(x), dim=1)
          return x
  ```,
  caption: [用于参照的卷积模型结构。]
) <code:conv-model>

== 训练与验证

我们使用学习率为0.0001的Adam @kingma2017adam 优化算法进行模型训练，所有模型均训练100轮。在每轮中，训练集的所有数据以$B=128$的批次输入模型进行训练。全部100轮完成后，统计模型的预测指标作为性能指标。

= 实验结果与分析

== 训练时收敛速度

如@<figure:training-loss> 所示，层数越多、参数量越大的模型在训练时准确率收敛的速度越慢。其中，带有4层线性层的模型起初准确率始终较低，后在约80轮处出现准确率的急速上升，后稳定在与其他同激活函数模型类似的性能。仅带有1层输出层的模型最终性能较差。

#img(
  image("assets/training-loss-acc.png"),
  caption: [模型训练时准确率与损失函数变化]
) <figure:training-loss>

== 二元预测性能

由@table:bin-result-label-6 和@table:bin-result-label-9 可以看出，在相同层数情况下，Tanh激活函数的召回率与$F_1$分数均最优，但是收到较低精确率的影响，整体性能（ROC AUC）最差。相同激活函数的模型中，模型越大，精确率越高，但召回率随参数数量增多先上升后下降。这可能是由于过大的模型容易过拟合训练数据，难以泛化。

#grid(
  columns: (1fr,) * 2,
  rows: (auto),
  gutter: 1em,
  [
    #img(
      image("assets/roc-label-6.png"),
      caption: [预测类别“6”的ROC曲线]
    ) <figure:roc-label-6>
  ],
  [
    #img(
      image("assets/roc-label-9.png"),
      caption: [预测类别“9”的ROC曲线]
    ) <figure:roc-label-9>
  ]
)

#tbl(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    toprule(),
    table.header([模型], [精确率], [召回率], [$F_1$分数], [ROC的AUC]),
    midrule(),
    [Baseline], [0.9769], [0.9697], [0.9733], [*0.9995*],
    [Tanh], [0.9751], [*0.9812*], [*0.9781*], [0.9992],
    [Sigmoid], [0.9678], [0.9718], [0.9698], [0.9993],
    [1 Layers], [0.9373], [0.9520], [0.9446], [0.9980],
    [2 Layers], [0.9738], [0.9718], [0.9728], [0.9994],
    [4 Layers], [0.9657], [0.9708], [0.9682], [0.9994],
    [5 Layers], [*0.9778*], [0.9676], [0.9727], [0.9988],
    bottomrule(),
  ),
  caption: "不同组合模型对分类“6”的预测性能",
) <table:bin-result-label-6>

#tbl(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    toprule(),
    table.header([模型], [精确率], [召回率], [$F_1$分数], [ROC的AUC]),
    midrule(),
    [Baseline], [0.9766], [0.9504], [0.9633], [0.9971],
    [Tanh], [*0.9818*], [*0.9633*], [*0.9725*], [*0.9991*],
    [Sigmoid], [0.9606], [0.9415], [0.9510], [0.9951],
    [1 Layers], [0.9173], [0.9009], [0.9090], [0.9889],
    [2 Layers], [0.9776], [0.9495], [0.9633], [0.9973],
    [4 Layers], [0.9542], [0.9495], [0.9518], [0.9965],
    [5 Layers], [0.9680], [0.9604], [0.9642], [0.9975],
    bottomrule(),
  ),
  caption: "不同组合模型对分类“9”的预测性能",
) <table:bin-result-label-9>

== 卷积模型与基线的性能对比

如@table:conv-baseline-perf-mnist 和@table:conv-baseline-perf-fashion 所示，在MNIST和FashionMNIST上，卷积模型均优于基线MLP模型。卷积操作具有空间不变性，在视觉任务上相比于MLP存在更大的优势。

#tbl(
  table(
    columns: 6,
    align: (center + horizon, center, center, center, center, center),
    toprule(),
    table.header([类别], [模型], [精确率], [召回率], [$F_1$分数], [ROC的AUC]),
    midrule(),
    table.cell(rowspan: 2, [“6”]), [Conv], [*0.9885*], [*0.9906*], [*0.9896*], [*0.9999*],
    [Baseline], [0.9769], [0.9697], [0.9733], [0.9995],
    midrule(),
    table.cell(rowspan: 2, [“9”]), [Conv], [*0.9823*], [*0.9891*], [*0.9857*], [*0.9998*],
    [Baseline], [0.9766], [0.9504], [0.9633], [0.9971],
    bottomrule(),
  ),
  caption: "卷积模型与基线模型在MNIST上的预测性能对比",
) <table:conv-baseline-perf-mnist>

#tbl(
  table(
    columns: 6,
    align: (center + horizon, center, center, center, center, center),
    toprule(),
    table.header([类别], [模型], [精确率], [召回率], [$F_1$分数], [ROC的AUC]),
    midrule(),
    table.cell(rowspan: 2, [“Shirt”]), [Conv], [*0.7651*], [*0.7330*], [*0.7487*], [*0.9660*],
    [Baseline], [0.7562], [0.5180], [0.6148], [0.9166],
    midrule(),
    table.cell(rowspan: 2, [“Ankle boot”]), [Conv], [*0.9727*], [*0.9620*], [*0.9673*], [*0.9986*],
    [Baseline], [0.9672], [0.9430], [0.9549], [0.9980],
    bottomrule(),
  ),
  caption: "卷积模型与基线模型在FashionMNIST上的预测性能对比",
) <table:conv-baseline-perf-fashion>

= 实验代码

本实验代码的Jupyter Notebook版本与本报告源码也可以从#link("https://github.com/CSharperMantle/hdu2025_deep_learning/tree/main/2-mlp")处获取。

```python
# %% [markdown]
# # Assignment 2: Simple multi layer perceptron

# %%
import functools as ft
import typing as ty

import matplotlib.pyplot as plt
import mnist as mnist_loader
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

np.random.seed(0x0D000721)
t.manual_seed(0x0D000721)

DPI = 200

# %% [markdown]
# ## 1. Algorithm
# 
# ### 1.1. Nonlinear functions
#
# * Rectified linear units (ReLU): $f_{\mathrm{ReLU}}(x) = \mathrm{max}\{0, x\}$
# * Sigmoid: $f_{\mathrm{sigmoid}}(x) = \frac{1}{1 + e^{-x}}$
# * Softmax: $f_{\mathrm{softmax}}(\bm{x})_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$
# * Tanh: $f_{\mathrm{tanh}}(x) = \mathrm{tanh}(x)$
#
# ### 1.2. Perceptron
#
# $$
# p(\bm{x}, \bm{A}, \bm{b}):\ \bm{y} = f_{\mathrm{act}}(\bm{x} \bm{A}^\intercal + \bm{b})
# $$
#
# where $f_{\mathrm{act}}$ is some nonlinear function ("activation function"). $\bm{A}$ and $\bm{b}$ are trainable parameters.
# 
# ### 1.3. Deeper networks
#
# Denote each layer function as $f_{\mathrm{layer}}(\bm{x}, \theta)$:
# 
# $$
# N(\bm{x}, \bm{\theta}):\ \bm{y} = \left(\underset{i \in I}{\mathop{\bigcirc}}\ f_{\mathrm{layer}\ i}\left(\cdot, \theta\right)\right)\left(\bm{x}\right)
# $$
# 
# ### 1.4. Supervised training
#
# Given observations $\bm{x}_s$ and respective truths $\bm{y}_s$, training is framed as the following optimization problem:
# 
# $$
# \hat{\bm{\theta}} = \underset{\bm{\theta}}{\arg\min}\ f_{\mathrm{loss}}\left(N\left(\bm{x}_s, \bm{\theta}\right), \bm{y}_s\right)
# $$
#
# where $f_{\mathrm{loss}}$ is some distance ("loss") defined on the truth space.

# %% [markdown]
# ## 2. Experiment setup

# %% [markdown]
# ### 2.1. Dataset preparation

# %%
with np.load("../dataset/mnist.npz") as mnist:
    mnist_x_train, mnist_y_train = mnist["x_train"], mnist["y_train"]
    mnist_x_test, mnist_y_test = mnist["x_test"], mnist["y_test"]

mnist_x_train = (
    mnist_x_train.reshape((mnist_x_train.shape[0], -1)).astype(np.float32) / 255.0
)
mnist_y_train = mnist_y_train.astype(np.int_)
mnist_x_test = (
    mnist_x_test.reshape((mnist_x_test.shape[0], -1)).astype(np.float32) / 255.0
)
mnist_y_test = mnist_y_test.astype(np.int_)

print(f"mnist_x_train.shape: {mnist_x_train.shape} dtype={mnist_x_train.dtype}")
print(f"mnist_y_train.shape: {mnist_y_train.shape} dtype={mnist_y_train.dtype}")
print(f"mnist_x_test.shape: {mnist_x_test.shape} dtype={mnist_x_test.dtype}")
print(f"mnist_y_test.shape: {mnist_y_test.shape} dtype={mnist_y_test.dtype}")

mnist_n_labels = max(int(mnist_y_train.max()), int(mnist_y_test.max())) + 1
print(f"mnist_n_labels = {mnist_n_labels}")
assert mnist_n_labels == 10

fashion_loader = mnist_loader.MNIST("../dataset/fashion_mnist", gz=True)
fashion_x_train, fashion_y_train = fashion_loader.load_training()
fashion_x_test, fashion_y_test = fashion_loader.load_testing()

fashion_x_train = np.array(fashion_x_train)
fashion_y_train = np.array(fashion_y_train)
fashion_x_test = np.array(fashion_x_test)
fashion_y_test = np.array(fashion_y_test)

fashion_x_train = (
    fashion_x_train.reshape((fashion_x_train.shape[0], -1)).astype(np.float32) / 255.0
)
fashion_y_train = fashion_y_train.astype(np.int_)
fashion_x_test = (
    fashion_x_test.reshape((fashion_x_test.shape[0], -1)).astype(np.float32) / 255.0
)
fashion_y_test = fashion_y_test.astype(np.int_)

print(f"fashion_x_train.shape: {fashion_x_train.shape} dtype={fashion_x_train.dtype}")
print(f"fashion_y_train.shape: {fashion_y_train.shape} dtype={fashion_y_train.dtype}")
print(f"fashion_x_test.shape: {fashion_x_test.shape} dtype={fashion_x_test.dtype}")
print(f"fashion_y_test.shape: {fashion_y_test.shape} dtype={fashion_y_test.dtype}")

fashion_n_labels = max(int(fashion_y_train.max()), int(fashion_y_test.max())) + 1
print(f"fashion_n_labels = {fashion_n_labels}")
assert fashion_n_labels == 10

# %% [markdown]
# ### 2.2. Baseline model structure
# 
# Our baseline model (append batch size $B$ in front of all shapes):
# 
# 1. FC1: 784 -> 128
# 2. ReLU1
# 3. FC2: 128 -> 64
# 4. ReLU2
# 5. FC3: 64 -> 10
# 6. Softmax

# %%
class MLPBaseline(nn.Module):
    def __init__(self):
        super(MLPBaseline, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

BASELINE_MODEL = MLPBaseline()

# %% [markdown]
# ### 2.3. Differential models
# 
# We mutate the models in activation functions and layer count.
# 
# * Activation: ReLU (baseline), Tanh, Sigmoid
# * Layer count: 1, 2, 3 (baseline), 4, 5

# %%
class MLPVarActivation(nn.Module):
    def __init__(self, activation):
        super(MLPVarActivation, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

VAR_ACT_MODELS: dict[ty.Literal["tanh", "sigmoid"], nn.Module] = {
    "tanh": MLPVarActivation(activation=F.tanh),
    "sigmoid": MLPVarActivation(activation=F.sigmoid),
}

class MLPVarLayer(nn.Module):
    def __init__(self, hidden_features: list[int]):
        super(MLPVarLayer, self).__init__()
        self.hidden_layers = nn.ModuleList()
        if len(hidden_features) > 0:
            self.hidden_layers.append(nn.Linear(784, hidden_features[0]))
            for i in range(1, len(hidden_features)):
                self.hidden_layers.append(nn.Linear(hidden_features[i - 1], hidden_features[i]))
            self.output_layer = nn.Linear(hidden_features[-1], 10)
        else:
            self.output_layer = nn.Linear(784, 10)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x

VAR_LAYER_MODELS: dict[ty.Literal[1, 2, 4, 5], nn.Module] = {
    1: MLPVarLayer(hidden_features=[]),
    2: MLPVarLayer(hidden_features=[128]),
    4: MLPVarLayer(hidden_features=[128, 64, 32]),
    5: MLPVarLayer(hidden_features=[256, 128, 64, 32]),
}

# %% [markdown]
# Another differential model is a simple convolution model.

# %%
class ConvClassifier(nn.Module):
    def __init__(self, p_dropout=0.2):
        super(ConvClassifier, self).__init__()
        # Input: (B, 784) -> (B, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # -> (B, 32, 24, 24)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5)  # -> (B, 16, 20, 20)
        self.fc1 = nn.Linear(16 * 20 * 20, 128)  # -> (B, 128)
        self.fc2 = nn.Linear(128, 10)  # -> (B, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 20 * 20)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


CONV_MODEL = ConvClassifier()

# %% [markdown]
# ### 2.4. Training and evaluation
# 
# We use the Adam (<https://arxiv.org/abs/1412.6980>) optimizer with $lr=0.0001$ for a fixed $100$ epochs. Each round, all training samples are passed forward in batches of $B=128$.

# %%
def train(
    model: nn.Module,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=100,
    batch_size=128,
    lr=0.0001,
) -> tuple[list[float], list[float]]:
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(t.from_numpy(x_train), t.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_test = t.from_numpy(x_test)
    y_test = t.from_numpy(y_test).long()

    acc_hist = []
    loss_hist = []

    with tqdm(total=epochs) as pbar:
        for _ in range(epochs):
            model.train()
            total_loss = 0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            model.eval()
            with t.inference_mode():
                test_pred = model(x_test).argmax(dim=1)
                test_acc = (test_pred == y_test).float().mean().item()

            acc_hist.append(test_acc)
            loss_hist.append(avg_loss)
            pbar.set_postfix(loss=avg_loss, acc=test_acc)
            pbar.update(1)

    return acc_hist, loss_hist

# %%
baseline_training_log = train(BASELINE_MODEL, mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test)
var_act_training_logs = {
    k: train(v, mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test)
    for k, v in VAR_ACT_MODELS.items()
}
var_layer_training_logs = {
    k: train(v, mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test)
    for k, v in VAR_LAYER_MODELS.items()
}

# %%
plt.close("all")

fig, ax = plt.subplots(1, 2, dpi=DPI)
fig.canvas.header_visible = False

# Plot accuracy curves
ax[0].plot(baseline_training_log[0], label='Baseline', linewidth=1.5)
for name, (acc_hist, _) in var_act_training_logs.items():
    ax[0].plot(acc_hist, label=f'{name.capitalize()}', linewidth=1.5)
for n_layers, (acc_hist, _) in var_layer_training_logs.items():
    ax[0].plot(acc_hist, label=f'{n_layers} layers', linewidth=1.5)

ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Test Accuracy')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Plot loss curves
ax[1].plot(baseline_training_log[1], label='Baseline', linewidth=1.5)
for name, (_, loss_hist) in var_act_training_logs.items():
    ax[1].plot(loss_hist, label=f'{name.capitalize()}', linewidth=1.5)
for n_layers, (_, loss_hist) in var_layer_training_logs.items():
    ax[1].plot(loss_hist, label=f'{n_layers} layers', linewidth=1.5)

ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_title('Training Loss')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.show()

# %% [markdown]
# ## 3. Evaluation

# %%
class BinPredEvalResult(ty.NamedTuple):
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def f1_score(self) -> float:
        prec = self.precision
        rec = self.recall
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def eval_bin_classif(
    model: nn.Module, x_test, y_test, target_class: int
) -> tuple[BinPredEvalResult, float]:
    assert 0 <= target_class <= 9
    model.eval()
    with t.inference_mode():
        yhat = model(t.from_numpy(x_test))
    yhat_bin = yhat.argmax(dim=1) == target_class
    y_bin = y_test == target_class
    yhat_scores = yhat[:, target_class].numpy()

    tp = (yhat_bin & y_bin).sum().item()
    fp = (yhat_bin & ~y_bin).sum().item()
    tn = (~yhat_bin & ~y_bin).sum().item()
    fn = (~yhat_bin & y_bin).sum().item()

    roc_auc = roc_auc_score(y_bin, yhat_scores)

    return BinPredEvalResult(tp=tp, fp=fp, tn=tn, fn=fn), roc_auc

# %% [markdown]
# ### 3.1. Precision, Recall and F1

# %%
LABELS = (6, 9)

bin_results: dict[int, pd.DataFrame] = {}
for label in LABELS:
    baseline_bin_result = eval_bin_classif(
        BASELINE_MODEL, mnist_x_test, mnist_y_test, target_class=label
    )
    var_act_bin_results = {
        k: eval_bin_classif(
            VAR_ACT_MODELS[k], mnist_x_test, mnist_y_test, target_class=label
        )
        for k in VAR_ACT_MODELS
    }
    var_layer_bin_results = {
        k: eval_bin_classif(
            VAR_LAYER_MODELS[k], mnist_x_test, mnist_y_test, target_class=label
        )
        for k in VAR_LAYER_MODELS
    }

    data = {
        "Baseline": {
            "Precision": baseline_bin_result[0].precision,
            "Recall": baseline_bin_result[0].recall,
            "F1": baseline_bin_result[0].f1_score,
            "AUC of ROC": baseline_bin_result[1]
        }
    }
    for k in var_act_bin_results:
        bin_metrics = var_act_bin_results[k][0]
        data[k.capitalize()] = {
            "Precision": bin_metrics.precision,
            "Recall": bin_metrics.recall,
            "F1": bin_metrics.f1_score,
            "AUC of ROC": var_act_bin_results[k][1]
        }
    for k in var_layer_bin_results:
        bin_metrics = var_layer_bin_results[k][0]
        data[f"{k} Layers"] = {
            "Precision": bin_metrics.precision,
            "Recall": bin_metrics.recall,
            "F1": bin_metrics.f1_score,
            "AUC of ROC": var_layer_bin_results[k][1]
        }
    df = pd.DataFrame(data).T
    bin_results[label] = df

# %%
print('Label "6"')
bin_results[6]

# %%
print('Label "9"')
bin_results[9]

# %% [markdown]
# ### 3.2. ROC curves

# %%
plt.close("all")

all_models: dict[str, nn.Module] = {
    "Baseline": BASELINE_MODEL,
}
all_models.update({k.capitalize(): v for k, v in VAR_ACT_MODELS.items()})
all_models.update({f"{k} Layers": v for k, v in VAR_LAYER_MODELS.items()})

for i, label in enumerate(LABELS):
    plt.figure(figsize=(8, 6), dpi=DPI)

    for model_name, model in all_models.items():
        model.eval()
        with t.inference_mode():
            yhat = model(t.from_numpy(mnist_x_test))

        y_bin = mnist_y_test == label
        yhat_score = yhat[:, label].numpy()

        fpr, tpr, _ = roc_curve(y_bin, yhat_score)
        auc = roc_auc_score(y_bin, yhat_score)

        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", linewidth=1.5)

    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5000)", linewidth=1.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f'Label "{label}"')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Comparison with convolution model

# %% [markdown]
# ### 4.1. MNIST

# %%
train(CONV_MODEL, mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test)

# %%
conv_fashion_results: dict[int, pd.Series] = {}

for k in LABELS:
    conv_result = eval_bin_classif(CONV_MODEL, mnist_x_test, mnist_y_test, k)
    data = {
        "Precision": conv_result[0].precision,
        "Recall": conv_result[0].recall,
        "F1": conv_result[0].f1_score,
        "AUC of ROC": conv_result[1],
    }
    conv_fashion_results[k] = pd.Series(data)

baseline_mnist_results = {k: bin_results[k].T["Baseline"] for k in bin_results}

conv_baseline_results = {
    k: pd.DataFrame(
        {"Conv": conv_fashion_results[k], "Baseline": baseline_mnist_results[k]}
    ).T
    for k in LABELS
}

# %%
conv_baseline_results[LABELS[0]]

# %%
conv_baseline_results[LABELS[1]]

# %% [markdown]
# ### 4.2. FashionMNIST

# %%
BASELINE_MODEL = MLPBaseline()
CONV_MODEL = ConvClassifier()

train(BASELINE_MODEL, fashion_x_train, fashion_y_train, fashion_x_test, fashion_y_test)
train(CONV_MODEL, fashion_x_train, fashion_y_train, fashion_x_test, fashion_y_test)

# %%
conv_baseline_results: dict[int, pd.DataFrame] = {}
for k in LABELS:
    baseline_result = eval_bin_classif(BASELINE_MODEL, fashion_x_test, fashion_y_test, k)
    conv_result = eval_bin_classif(CONV_MODEL, fashion_x_test, fashion_y_test, k)
    data = {
        "Conv": {
            "Precision": conv_result[0].precision,
            "Recall": conv_result[0].recall,
            "F1": conv_result[0].f1_score,
            "AUC of ROC": conv_result[1],
        },
        "Baseline": {
            "Precision": baseline_result[0].precision,
            "Recall": baseline_result[0].recall,
            "F1": baseline_result[0].f1_score,
            "AUC of ROC": baseline_result[1],
        },
    }
    conv_baseline_results[k] = pd.DataFrame(data).T

# %%
conv_baseline_results[LABELS[0]]

# %%
conv_baseline_results[LABELS[1]]
```

#pagebreak()

#bibliography("bib.bib", style: "gb-7714-2015-numeric")
