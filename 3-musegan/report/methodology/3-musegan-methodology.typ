#import "@preview/ouset:0.2.0": underset
#import "@preview/booktabs:0.0.4": *

#import "../../../template/hdu-report-typst/template/template.typ": *

#show: booktabs-default-table-style
#show: project.with(
  title: "《深度学习》期末作业",
  subtitle: [方法研究：基于 MuseGAN 的 MIDI 片段生成],
  class: "计算机科学英才班",
  department: "卓越学院",
  authors: "鲍溶",
  author_id: "23060827",
  date: datetime(year: 2025, month: 12, day: 07),
  cover_style: "hdu_report",
)

#show link: underline

#toc()

#pagebreak()

= 模型结构

#img(
  image("assets/gan-outline.png", width: 75%),
  caption: [MuseGAN 的对抗模型]
) <figure:gan-outline>

如@figure:gan-outline，MuseGAN 的模型由两个主要部件构成：生成模块 $G$ 从随机噪声中生成乐谱数据，鉴别模块 $D$ 尝试将生成的乐谱数据与真实训练数据进行二分类 @dong2018musegan。对 $D$ 的训练能够促进 $G$ 产生更为符合真实乐谱性质的输出，从而起到对抗的效果。

== 鉴别模块

#img(
  image("assets/d-outline.png", width: 50%),
  caption: [鉴别模块的结构]
) <figure:d-outline>

如图@figure:d-outline，鉴别模块对于每个 $(G(z), x)$ 二元组计算三种损失：分类 $G(z)$ 的 Wasserstein 距离（即精确率的相反数）、分类 $x$ 的 Wasserstein 距离、分类 $0.5 G(z) + 0.5x$ 的梯度惩罚值。最后一项损失可以有效防止鉴别模块产生过拟合。图中 $C$ 单元结构如@figure:critic。

#img(
  image("assets/critic.png"),
  caption: [$C$ 的结构]
) <figure:critic>

== 生成模块

#img(
  image("assets/g-outline.png", width: 75%),
  caption: [生成模块的结构]
) <figure:g-outline>

#img(
  image("assets/temporal-net.png"),
  caption: [$G_#text([temp])$ 的结构]
) <figure:temporal-net>

#img(
  image("assets/bar-gen.png"),
  caption: [$G_#text([bar])$ 的结构]
) <figure:bar-gen>

生成模块由多个子网络组成，共同完成从随机噪声到多轨音乐片段的生成。整体结构如@figure:g-outline 所示，生成器接收四种噪声输入：和弦（chords）、风格（style）、旋律（melody）和节奏（groove）。其中和弦和风格为全局共享，旋律和节奏则针对每个音轨独立生成。

和弦生成器和旋律生成器均为时态网络，其结构如@figure:temporal-net 所示。时态网络通过转置卷积将一维噪声向量扩展为二维特征图，最终输出形状为 $(B, Z, N_B)$ 的时态特征序列。每个音轨的时态特征与和弦、风格、节奏特征拼接后，送入对应的小节生成器。

小节生成器结构如@figure:bar-gen 所示，采用全连接层和转置卷积层的级联结构。输入的四维噪声向量首先通过全连接层映射到高维空间，再经过四次转置卷积操作逐步上采样，最终输出单条音乐数据。所有音轨和所有小节的输出拼接后，形成完整的 $(B, N_T, N_B, N_S, N_P)$ 五维张量。

= 实验方法

== 训练

实验方法采用 Wasserstein GAN with Gradient Penalty（WGAN-GP）训练框架。训练阶段，鉴别器每轮迭代更新 5 次，生成器更新 1 次，使用 Adam 优化器（学习率0.001，$beta_1 = 0.5$，$beta_2 = 0.9$）。数据集采用 LPD-5 乐谱数据集，为加速训练过程，实际使用数据量缩减至原数据集的95%。每轮训练后保存模型检查点，并记录生成器和鉴别器的损失函数变化。

== 验证

验证阶段采用双重评估策略。定量评估通过计算生成样本与真实样本在特征空间的分布距离，使用 Fréchet Inception Distance（FID）指标。定性评估通过将生成的五维张量后处理为 MIDI 文件，人工聆听评估音乐的结构连贯性、旋律合理性和节奏稳定性。生成样本的和声进行、音高分布等音乐特性与训练数据统计特征进行对比分析。

== 计算量评估

计算量估计基于 `torchinfo` 工具 @yep2020torchinfo 对模型各组件进行分析。时态网络参数量为 1083904，计算量为 33.56M 次乘法加法操作。小节生成器参数量为 3559424，计算量为 108.42M 次乘法加法。完整生成模块参数量为 23036928，计算量为 701.15M 次乘法加法。鉴别模块参数量为 1024000，计算量为 31.26M 次乘法加法。

#pagebreak()

#bibliography("bib.bib", style: "gb-7714-2015-numeric")
