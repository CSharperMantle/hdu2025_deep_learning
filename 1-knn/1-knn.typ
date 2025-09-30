#import "@preview/algorithmic:1.0.5"
#import algorithmic: algorithm-figure, style-algorithm
#import "@preview/booktabs:0.0.4": *

#import "../template/hdu-report-typst/template/template.typ": *

#show: style-algorithm
#show: booktabs-default-table-style
#show: project.with(
  title: "《深度学习》作业1",
  subtitle: [基于$k$-近邻分类模型的MNIST图像分类],
  class: "计算机科学英才班",
  department: "卓越学院",
  authors: "鲍溶",
  author_id: "23060827",
  date: (2025, 09, 26),
  cover_style: "hdu_report",
)

#show link: underline

#toc()
#pagebreak()

= 方法简介

== $k$-近邻分类模型

$k$-近邻分类通过求取给定查询元素的$k$个最近邻居确定该元素在样本空间中的可能归属，所有求得邻居元素中类别的众数即为预测分类。

#algorithm-figure(
  [$k$-近邻分类],
  supplement: "算法",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      [$k$-Nearest-Neighbors-Classify],
      ([$bold(x)$], [$S$], [$k$]),
      {
        Comment[$bold(x)$ #h(1em) 查询元素]
        Comment[$S$ #h(1em) 样本元素集合]
        Comment[$k$ #h(1em) 考虑的邻居个数]
        LineBreak
        Assign[$D$][${ ||bold(x)-bold(s)|| : bold(s) in S }$]
        Assign[$N$][first $k$ closest members from $D$]
        Return[$bold("mode")(N)."label"$]
      },
    )
  },
) <algorithm:knn>

该算法的朴素实现为迭代全部样本元素，求得与查询元素距离后进行排序，求取最小$k$个元素，再求得众数。scikit-learn @scikit-learn 的实现利用了$k$-D树和球树等数据结构，以加速求取最小元素的过程。

== MNIST数据集

MNIST数据集为经典的黑白手写数字数据集，包含60000个训练样本与10000个测试样本。每个样本点为$28 times 28 = 784$维向量，量化至8位整型空间。

== 分类模型性能指标

分类模型性能评价主要涉及以下指标：

- *准确率*：正确分类的实例占总实例的比例。计算公式为$C = ("TP" + "TN")/("TP" + "FP" + "TN" + "FN")$，其中$"TP"$表示真正例，$"TN"$表示真负例，$"FP"$表示假正例，$"FN"$表示假负例。
- *精确率*：在所有被预测为正例的实例中，真正为正例的比例，计算公式为$P = ("TP")/("TP" + "FP")$。该指标反映了模型正例预测的可靠性。
- *召回率*：也称为敏感度，表示所有实际正例中被正确识别的比例，计算公式为$R = ("TP")/("TP" + "FN")$。该指标反映了模型发现所有正例的能力。
- *$F_1$分数*：精确率和召回率的调和平均数，计算公式为$F_1 = 2 times (P times R)/(P + R)$，提供了两个指标之间的平衡。
- *ROC曲线与AUC*：接收者操作特征（ROC）曲线由不同阈值下真正例率对假正例率的关系表格绘制。其曲线下面积（AUC）由积分得到，越接近1表示曲线越贴近左上角，模型性能越好。

对于多分类问题通常使用一对多或一对一的方式计算这些指标。

= 实验设置

== 数据集预处理

我们将原始的$SS = ZZ_256^784$空间映射到$SS^* = [0, 1]^784$空间对数据集进行归一化预处理，借助$SS^*$空间的$ell_2$范数计算两归一化样本之间的距离。

== 实验环境与方法

我们使用Python 3.13与scikit-learn @scikit-learn、numpy @harris2020array、matplotlib @matplotlib 等库完成本实验。

出于性能考虑，我们首先使用朴素方法实现$k$-近邻算法，并以scikit-learn提供的`KNeighborsClassifier`作为参考实现，使用MNIST上随机抽取的少数测试样本验证朴素算法的正确性。在之后的步骤中，将使用`KNeighborsClassifier`作为对象，在完整MNIST数据集上进行实验。

我们首先研究邻居数量$k$对预测准确率的影响，之后尝试通过主成分分析法提取样本中的主要特征，并研究不同降维因子在相同$k$值下对准确率的影响。之后，选取两个分类“6”与“9”，将问题简化为两个二元（正负）分类问题，分别研究器准确率$C$、精确率$C$、召回率$R$、$F_1$分数、ROC曲线及其AUC。最后，考虑所有10个分类，研究算法的多元预测准确率、精确率与召回率。

== 参数设置

在研究邻居数量$k$对预测准确率的影响时，在${1, 3, ..., 99}$中选取$k$值。

在研究特征提取对预测性能影响时，固定$k=3$，在${1, 2, ..., 31}$中选取降维因子$f$，最终样本$bold(s)' in [0, 1]^(floor(784 / f))$。

在研究准确率、精确率与召回率时，固定$k=3$；在研究ROC曲线与其AUC时，固定$k=31$。

= 实验结果与分析

== 实现验证

#code(
  ```python
  def naive_knn(
      x: np.ndarray,
      samples: np.ndarray,
      labels: np.ndarray,
      n_labels: int,
      k: int,
  ) -> tuple[int, np.ndarray]:
      dists = np.linalg.norm(samples - x, ord=2, axis=1)
      kni = np.argsort(dists)[:k]
      counts = np.bincount(labels[kni], minlength=n_labels)
      return int(np.argmax(counts)), (counts / k)
  ```,
  caption: [用Python实现的朴素$k$-近邻算法],
) <code:knn>

如@code:knn 所示的实现能够通过100个随机样本点的正确性测试。在Core i9-13900HX平台上，以MNIST训练数据为样本元素集合，可以达到2.86秒每查询的速度。

== 近邻数量选择

#img(
  image("assets/k-vs-c.png"),
  caption: "近邻数与准确率关系图",
) <figure:k-vs-c>

如@figure:k-vs-c 所示，随着近邻数量增加，分类模型准确率先上升后下降，最高点在$k=3$时取到。这可能由于在分类时考虑太多邻居会引入额外的噪声，降低分类模型的准确率。

== 特征提取

#img(
  image("assets/reduction-vs-c.png"),
  caption: [特征降维因子与准确率关系图（$k=3$）],
) <figure:reduction-vs-c>

如@figure:reduction-vs-c 所示，随着样本维度降低，分类模型准确率先上升后下降，在降低至初始维数的$1/13$（即60维）时准确率达到最高，随后震荡下降。由于原样本为图像直接归一化得到，具有较多冗余信息，对其进行特征提取有助于同类别样本点在空间中靠拢，帮助分类模型提升准确率。同时，由于缩减维数会丢失信息，过度降维会使分类模型难以得到有用信息。

== 二元分类性能分析

观察@figure:bin-class-confusion 和@table:bin-class-perf 可以发现，$k=3$时分类模型能达到较为良好的正负分类效果，其准确率、精确率、召回率、$F_1$分数均较为接近1。在$k=31$时测量得到如@figure:bin-class-r-vs-p 所示的精确率--召回率曲线，可以观察得到仅在召回率较为接近1时，精确率才开始下降。

#tbl(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    toprule(),
    table.header([类别], [准确率], [精确率], [召回率], [$F_1$分数]),
    midrule(),
    [“6”], [0.9970], [0.9854], [0.9833], [0.9844],
    [“9”], [0.9919], [0.9594], [0.9603], [0.9598],
    bottomrule(),
  ),
  caption: "两个二分类任务的性能指标",
) <table:bin-class-perf>

#img(
  image("assets/bin-class-confusion.png", width: 90%),
  caption: [分类“6”与分类“9”的混淆矩阵（$k=3$）],
) <figure:bin-class-confusion>

#img(
  image("assets/bin-class-r-vs-p.png"),
  caption: [分类“6”与分类“9”的$P$-$R$曲线（$k=31$）],
) <figure:bin-class-r-vs-p>

观察@table:bin-class-auc 与@figure:bin-class-roc 中的数据可发现，在$k=31$时，该分类模型在两个类别上的ROC曲线均极为贴近左上角，其AUC也均接近1，表明其在这两个标签下均有较好的预测性能。

#tbl(
  table(
    columns: 2,
    align: (center, center),
    toprule(),
    table.header([类别], [AUC]),
    midrule(), [“6”],
    [0.9981], [“9”],
    [0.9976], bottomrule(),
  ),
  caption: "不同二分类任务ROC曲线的AUC",
) <table:bin-class-auc>

#img(
  image("assets/bin-class-roc.png"),
  caption: [分类“6”与分类“9”的ROC曲线（$k=31$）],
) <figure:bin-class-roc>

== 多元分类性能分析

@figure:nary-class-confusion 所示的混淆矩阵中主对角线颜色均较深，而其他元素均较浅，说明该分类模型对于所有分类均有较好的性能。观察到模型较为频繁地将“7”错误识别为“1”、将“4”错误识别为“9”，这些字符对在形状上符合日常经验。该矩阵并不对称，说明错误识别并不存在交换关系。

@table:nary-class-perf 显示，模型对于所有类别均有较好的识别性能。模型对于“9”类别的识别准确率最低，对“6”类别的识别准确率最高。

#tbl(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    toprule(),
    table.header([类别], [准确率], [精确率], [召回率], [$F_1$分数]),
    midrule(), [“0”], [0.9960], [0.9939], [0.9663],
    [0.9799], [“1”], [0.9948], [0.9982], [0.9577],
    [0.9776], [“2”], [0.9946], [0.9651], [0.9822],
    [0.9736], [“3”], [0.9929], [0.9663], [0.9635],
    [0.9649], [“4”], [0.9944], [0.9674], [0.9754],
    [0.9714], [“5”], [0.9937], [0.9630], [0.9663],
    [0.9646], [“6”], [0.9970], [0.9854], [0.9833],
    [0.9844], [“7”], [0.9927], [0.9640], [0.9649],
    [0.9645], [“8”], [0.9930], [0.9384], [0.9892],
    [0.9631], [“9”], [0.9919], [0.9594], [0.9603],
    [0.9598], bottomrule(),
  ),
  caption: "多分类任务各标签的性能指标",
) <table:nary-class-perf>

#img(
  image("assets/nary-class-confusion.png", width: 75%),
  caption: [全部10个分类的多元分类混淆矩阵（$k=3$）],
) <figure:nary-class-confusion>

= 实验代码

本实验代码的Jupyter Notebook版本与本报告源码也可以从#link("https://github.com/CSharperMantle/hdu2025_deep_learning/tree/main/1-knn")处获取。

```python
# %% [markdown]
# # Assignment 1: $k$-nearest neighbors

# %%
import functools as ft
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as RefKNeighborsClassifier
from tqdm.notebook import tqdm

np.random.seed(0x0d000721)

DPI = 200

# %% [markdown]
# ## 1. Algorithm
# 
# function k_nearest_neighbors(x, samples, k, distance)
#     dists <- sorted([distance(x, e) for e in samples])
#     neighbors <- first k samples with closest distances
#     return dominating label within neighbors
# endfunction

# %%
def naive_knn(
    x: np.ndarray,
    samples: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    k: int,
) -> tuple[int, np.ndarray]:
    dists = np.linalg.norm(samples - x, ord=2, axis=1)
    kni = np.argsort(dists)[:k]
    counts = np.bincount(labels[kni], minlength=n_labels)
    return int(np.argmax(counts)), (counts / k)

# %% [markdown]
# ## 2. Experiment setup

# %%
with np.load("../dataset/mnist.npz") as mnist:
    x_train, y_train = mnist["x_train"], mnist["y_train"]
    x_test, y_test = mnist["x_test"], mnist["y_test"]

x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32) / 255.0
y_train = y_train.astype(np.intp)
x_test = x_test.reshape((x_test.shape[0], -1)).astype(np.float32) / 255.0
y_test = y_test.astype(np.intp)

print(f"x_train.shape: {x_train.shape} dtype={x_train.dtype}")
print(f"y_train.shape: {y_train.shape} dtype={y_train.dtype}")
print(f"x_test.shape: {x_test.shape} dtype={x_test.dtype}")
print(f"y_test.shape: {y_test.shape} dtype={y_test.dtype}")

n_labels = max(int(y_train.max()), int(y_test.max())) + 1
print(f"n_labels = {n_labels}")
assert n_labels == 10

# %% [markdown]
# ## 3. Verification
# 
# We use the `KNeighborsClassifier` provided by scikit-learn as the reference implementation.

# %%
N_VERIF = 100

verif_samples = x_test[np.random.choice(x_test.shape[0], size=N_VERIF, replace=False)]
classifier = RefKNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(x_train, y_train)
for sample in tqdm(verif_samples):
    sample = sample[np.newaxis, :]
    pred_ref = classifier.predict(sample).squeeze(0)
    pred_ref_prob = classifier.predict_proba(sample).squeeze(0)
    pred_uut, pred_uut_prob = naive_knn(sample[0], x_train, y_train, 10, 5)
    assert pred_ref == pred_uut, f"pred_ref={pred_ref}, pred_uut={pred_uut}"
    assert np.allclose(
        pred_ref_prob, pred_uut_prob
    ), f"pred_ref_prob={pred_ref_prob}, pred_uut_prob={pred_uut_prob}"

# %% [markdown]
# ## 4. Evaluation
# 
# For prediction performance, we evaluate combinations of:
# 
# * Different values for $k$
# * With or without feature extraction
# 
# For runtime efficiency, we evaluate how dimensionality of the sample vector influence performance.
# 
# We use the following metrics for binary prediction evaluation:
# 
# * Accuracy $C$, recall rate $R$, precision $P$
# * $F_1$ score
# * ROC curve and AUC
# 
# We then evaluate multi-class prediction by $R$ and $P$.

# %% [markdown]
# ### 4.1. $k$'s influence on performance

# %%
c_against_k: list[tuple[int, float]] = []
for k in tqdm(range(1, 100, 2)):
    classifier = RefKNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)
    classifier.fit(x_train, y_train)
    neighbors = classifier.predict(x_test)
    correct = neighbors == y_test
    c_against_k.append((k, correct.sum() / x_test.shape[0]))

# %%
plt.close("all")
fig, ax = plt.subplots(dpi=DPI)
fig.canvas.header_visible = False

fac_vec = [k for k, _ in c_against_k]
c_vec = [c for _, c in c_against_k]
c_max, c_min = max(c_vec), min(c_vec)
ax.plot(fac_vec, c_vec, marker="o", linestyle="-", color="blue", markersize=4)
ax.set_ylim(top=min(1.0, c_max + 0.001), bottom=max(0.0, c_min - 0.001))
ax.set_xlabel("Number of neighbors ($k$)", fontsize=12)
ax.set_ylabel("Accuracy ($C$)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2. Feature extraction with PCA
# 
# We use Principal Component Analysis tool provided by scikit-learn to reduce the dimensionality of the sample vector. We use $k=3$ here.

# %%
K = 3

orig_n_dims = x_train.shape[1]
assert x_test.shape[1] == orig_n_dims

fac_acc: list[tuple[int, float]] = []
for fac in tqdm(range(1, 32)):
    n_dims = orig_n_dims // fac
    pca = PCA(n_components=n_dims)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    classifier = RefKNeighborsClassifier(n_neighbors=K, metric="minkowski", p=2)
    classifier.fit(x_train_pca, y_train)
    neighbors = classifier.predict(x_test_pca)
    correct = neighbors == y_test
    accuracy = correct.sum() / x_test.shape[0]
    fac_acc.append((fac, accuracy))

# %%
plt.close("all")
fig, ax = plt.subplots(dpi=DPI)
fig.canvas.header_visible = False

fac_vec = [fac for fac, _ in fac_acc]
c_vec = [c for _, c in fac_acc]
c_max, c_min = max(c_vec), min(c_vec)
ax.plot(fac_vec, c_vec, marker="o", linestyle="-", color="blue", markersize=4)
ax.set_ylim(top=min(1.0, c_max + 0.001), bottom=max(0.0, c_min - 0.001))
ax.set_xlabel("Reduction factor", fontsize=12)
ax.set_ylabel("Accuracy ($C$)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3. Binary prediction characteristics
# 
# We choose labels "6" and "9" for the following evaluation.

# %%
LABELS = (6, 9)

# %% [markdown]
# #### 4.3.1. $C$, $P$ and $R$ when $k=3$

# %%
K = 3


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


x_train_bin = {label: x_train.copy() for label in LABELS}
y_train_bin = {label: (y_train == label).astype(y_train.dtype) for label in LABELS}
x_test_bin = {label: x_test.copy() for label in LABELS}
y_test_bin = {label: (y_test == label).astype(y_test.dtype) for label in LABELS}

# %%
pred_bin: dict[int, BinPredEvalResult] = {}
for label in tqdm(LABELS):
    classifier = RefKNeighborsClassifier(n_neighbors=K, metric="minkowski", p=2)
    classifier.fit(x_train_bin[label], y_train_bin[label])
    neighbors = classifier.predict(x_test_bin[label])
    correct = neighbors == y_test_bin[label]
    accuracy = correct.sum() / x_test_bin[label].shape[0]
    tp = np.logical_and(neighbors == 1, y_test_bin[label] == 1).sum()
    fp = np.logical_and(neighbors == 1, y_test_bin[label] == 0).sum()
    tn = np.logical_and(neighbors == 0, y_test_bin[label] == 0).sum()
    fn = np.logical_and(neighbors == 0, y_test_bin[label] == 1).sum()
    result = BinPredEvalResult(tp=tp, fp=fp, tn=tn, fn=fn)
    pred_bin[label] = result
    print(
        f'Label "{label}":\tC={result.accuracy:.4f}\tR={result.recall:.4f}\tP={result.precision:.4f}\tF1={result.f1_score:.4f}'
    )

# %%
plt.close("all")

fig, ax = plt.subplots(1, 2, dpi=DPI)
fig.canvas.header_visible = False

for i, label in enumerate(LABELS):
    result = pred_bin[label]
    cm = np.array([[result.tp, result.fn], [result.fp, result.tn]])
    cm_scaled = np.log(cm + 1)
    im = ax[i].imshow(cm_scaled, interpolation="nearest", cmap=plt.cm.Blues)

    thresh = cm_scaled.max() / 2
    for j in range(2):
        for k in range(2):
            ax[i].text(
                k,
                j,
                f"{cm[j, k]}",
                ha="center",
                va="center",
                color="white" if cm_scaled[j, k] > thresh else "black",
            )

    ax[i].set_title(f'Label "{label}"')
    ax[i].set_xticks([0, 1])
    ax[i].set_yticks([0, 1])
    ax[i].set_xticklabels(["1", "0"])
    ax[i].set_yticklabels(["1", "0"])
    ax[i].set_xlabel("Prediction")
    if i == 0:
        ax[i].set_ylabel("Truth")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### 4.3.2. P-vs-R curve, ROC, and its AUC when $k=31$

# %%
K = 31

conf_bin: dict[int, npt.NDArray[np.float32]] = {}
for label in tqdm(LABELS):
    classifier = RefKNeighborsClassifier(n_neighbors=K, metric="minkowski", p=2)
    classifier.fit(x_train_bin[label], y_train_bin[label])

    pred = classifier.predict(x_test_bin[label])
    neighbors = classifier.kneighbors(x_test_bin[label], return_distance=False)
    assert isinstance(neighbors, np.ndarray)
    assert neighbors.shape == (x_test_bin[label].shape[0], K)

    confidences = np.zeros((x_test_bin[label].shape[0],), dtype=np.float32)
    for i in range(x_test_bin[label].shape[0]):
        counts = np.bincount(y_train_bin[label][neighbors[i]], minlength=2)
        confidences[i] = counts[1] / K
    assert np.all(confidences >= 0.0) and np.all(confidences <= 1.0)

    conf_bin[label] = confidences

# %%
N_STEPS = 101

plt.close("all")

fig, ax = plt.subplots(1, 2, dpi=DPI)
fig.canvas.header_visible = False

for i, label in enumerate(LABELS):
    y_true = y_test_bin[label]
    y_scores = conf_bin[label]
    p = y_true.sum()
    n = y_true.shape[0] - p
    thresholds = np.linspace(-0.001, 1.001, num=N_STEPS, endpoint=True)
    tpr = []
    fpr = []
    for j, thr in enumerate(thresholds):
        y_pred = (y_scores >= thr).astype(y_true.dtype)
        tp = np.logical_and(y_pred == 1, y_true == 1).sum()
        fp = np.logical_and(y_pred == 1, y_true == 0).sum()
        tpr.append(tp / p if p > 0 else 0.0)
        fpr.append(fp / n if n > 0 else 0.0)
    tpr = np.array(tpr, dtype=np.float32)
    fpr = np.array(fpr, dtype=np.float32)

    print(f"Label {label}:\tAUC = {-np.trapezoid(tpr, fpr):.4f}")

    ax[i].plot(fpr, tpr, linestyle="-", color="blue")
    ax[i].set_title(f'Label "{label}"')
    ax[i].grid(True, linestyle="--", alpha=0.7)
    ax[i].set_xlim(left=0.0, right=1.0)
    ax[i].set_ylim(bottom=0.0, top=1.0)
    ax[i].set_xlabel("FPR")
    ax[i].set_aspect("equal", adjustable="box")
    if i == 0:
        ax[i].set_ylabel("TPR")

plt.tight_layout()
plt.show()

# %%
N_STEPS = 101

plt.close("all")

fig, ax = plt.subplots(1, 2, dpi=DPI)
fig.canvas.header_visible = False

for i, label in enumerate(LABELS):
    y_true = y_test_bin[label]
    y_scores = conf_bin[label]
    p = y_true.sum()
    n = y_true.shape[0] - p
    thresholds = np.linspace(-0.001, 1.001, num=N_STEPS, endpoint=True)
    recall = []
    prec = []
    for j, thr in enumerate(thresholds):
        y_pred = (y_scores >= thr).astype(y_true.dtype)
        tp = np.logical_and(y_pred == 1, y_true == 1).sum()
        fp = np.logical_and(y_pred == 1, y_true == 0).sum()
        tn = np.logical_and(y_pred == 0, y_true == 0).sum()
        fn = np.logical_and(y_pred == 0, y_true == 1).sum()
        result = BinPredEvalResult(tp=tp, fp=fp, tn=tn, fn=fn)
        recall.append(result.recall)
        prec.append(result.precision)
    recall = np.array(recall, dtype=np.float32)[:-1]
    prec = np.array(prec, dtype=np.float32)[:-1]

    ax[i].plot(recall, prec, linestyle="-", color="blue")
    ax[i].set_title(f'Label "{label}"')
    ax[i].grid(True, linestyle="--", alpha=0.7)
    ax[i].set_xlim(left=0.0, right=1.0)
    ax[i].set_ylim(bottom=0.0, top=1.0)
    ax[i].set_xlabel("Recall")
    ax[i].set_aspect("equal", adjustable="box")
    if i == 0:
        ax[i].set_ylabel("Precision")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4. Multi-class prediction characteristics
# 
# We assume $k=3$ in this section.

# %%
K = 3

classifier = RefKNeighborsClassifier(n_neighbors=K, metric="minkowski", p=2)
classifier.fit(x_train, y_train)
pred_nary = classifier.predict(x_test)

cm_nary = np.zeros((n_labels, n_labels), dtype=np.int32)

for true_label, pred_label in zip(y_test, pred_nary):
    cm_nary[true_label, pred_label] += 1

for i in range(n_labels):
    tp = cm_nary[i, i]
    fp = cm_nary[:, i].sum() - tp
    fn = cm_nary[i, :].sum() - tp
    tn = cm_nary.sum() - (tp + fp + fn)
    result = BinPredEvalResult(tp=tp, fp=fp, tn=tn, fn=fn)
    print(
        f"Label {i}:\tC={result.accuracy:.4f}\tR={result.recall:.4f}\tP={result.precision:.4f}\tF1={result.f1_score:.4f}"
    )

# %%
plt.close("all")

fig, ax = plt.subplots(dpi=DPI)
fig.canvas.header_visible = False

cm_nary_scaled = np.log(cm_nary + 1)

im = ax.imshow(cm_nary_scaled, interpolation="nearest", cmap=plt.cm.Blues)

thresh = cm_nary_scaled.max() / 2
for i in range(n_labels):
    for j in range(n_labels):
        ax.text(
            j,
            i,
            f"{cm_nary[i, j]}",
            ha="center",
            va="center",
            color="white" if cm_nary_scaled[i, j] > thresh else "black",
        )

ax.set_xticks(list(range(n_labels)))
ax.set_yticks(list(range(n_labels)))
ax.set_xticklabels([f'"{i}"' for i in range(n_labels)])
ax.set_yticklabels([f'"{i}"' for i in range(n_labels)])
ax.set_xlabel("Prediction")
ax.set_ylabel("Truth")

plt.tight_layout()
plt.show()
```

#pagebreak()

#bibliography("bib.bib", style: "gb-7714-2015-numeric")
