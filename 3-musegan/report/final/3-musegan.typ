#import "@preview/ouset:0.2.0": underset
#import "@preview/booktabs:0.0.4": *

#import "../../../template/hdu-report-typst/template/template.typ": *

#show: booktabs-default-table-style
#show: project.with(
  title: "《深度学习》期末作业",
  subtitle: [基于 MuseGAN 的 MIDI 片段生成],
  class: "计算机科学英才班",
  department: "卓越学院",
  authors: "鲍溶",
  author_id: "23060827",
  date: datetime(year: 2026, month: 01, day: 07),
  cover_style: "hdu_report",
)

#show link: underline

#align(center)[
  #text(weight: "bold", size: 14pt)[摘要]
]

MuseGAN @dong2018musegan 是一类面向符号化音乐的多轨生成对抗网络模型。与图像生成不同，符号化音乐同时具有显著的时间层次结构以及多声部、多乐器之间的协同依赖关系。本文使用 PyTorch 对 MuseGAN 的核心思想与工程实现进行复现，包括数据表示、生成器的时序结构、小节生成器组合方式、鉴别器的三维卷积判别结构，以及基于 Wasserstein 距离与梯度惩罚的对抗式训练流程。本文给出了实验中训练脚本与推理脚本的可复现运行方式、关键超参数、模型检查点与生成样例文件，并讨论了当前实现与原论文设定之间可能存在的差异与改进方向。

*关键词*：MuseGAN；生成对抗网络；Wasserstein GAN；多轨音乐生成；MIDI

#toc()

#pagebreak()

= 研究背景

生成对抗网络（GAN）在图像、视频、文本等生成任务中已经形成较为成熟的方法体系；但在符号化音乐生成任务中仍存在显著差异：

- *时间性与层次结构*：“音乐是时间的艺术。”@dong2018musegan 音乐在多个时间尺度上具有结构，如节拍、小节、乐句等，需要模型在局部连贯与全局结构之间取得平衡；
- *多轨依赖性*：非平凡的音乐通常由多个乐器轨道组成，如交响乐中的不同声部或乐器。这些轨道在时间上不仅各自发展，而且存在紧密的互动和依赖，构成音乐的整体；
- *复调与和声*：音乐中的音符通常组合成和弦或旋律，在线性时间轴上加入了第二个维度，使得在自然语言处理中常用的按时间顺序引入音符的方法并不完全适合。

MuseGAN @dong2018musegan 在上述背景下提出针对多轨钢琴卷轴生成的 GAN 框架，并在 Lakh Pianoroll Dataset 的子集上展示了多轨协同生成能力。其核心意义在于攻克多乐器协同生成的难点。论文中提出的“混合模型”结合了独立生成与统一生成的思想，利用共享的潜在向量来协调不同轨道的生成，从而模拟人类作曲家在编曲时对和声与整体结构的把控，对于提升生成音乐的听感和谐度有重要意义。

原文中，作者还提出了多种用于量化生成音乐片段质量的指标，如空小节（empty bars，EB）、音名丰度（used pitch classes per bar，UPC）、有效音符（ratio of qualified notes，QN，即长于 32 分的音符占比）、鼓谱（drum pattern，DP，8 或 16 拍子音符模式的占比）、调性距离（tonal distance，TD）@harte2006tonaldistance 等。这些指标在研究模型的内部结构、训练模型、筛选模型产物中起到了关键作用，也为之后的音频生成研究提供了参考。

= 研究目的

本次研究旨在复现 MuseGAN 论文中的核心模型，并利用 Lakh Pianoroll Dataset @raffel2016million 进行训练和评估，并以此加深对深度学习原理、模型构建、PyTorch 框架使用与科学研究方法的了解与运用。

= 研究方法

== 数据表示

MuseGAN 采用钢琴卷轴（piano-roll）作为符号化音乐的张量表示。结合本项目的数据准备脚本与数据集读取逻辑，训练样本张量的维度约定为：

$ x in RR^(N_T times N_B times N_S times N_P) $

其中：

- $N_T$：轨道数（tracks），本实现使用 5 条轨道；
- $N_B$：小节数（bars），本实现生成 4 小节；
- $N_S$：每小节的时间步数（steps per bar），本实现为 48；
- $N_P$：音高维度（pitches），本实现为 84。

训练时再加上 Batch 维度，形成 $RR^(B times N_T times N_B times N_S times N_P)$ 的五维张量。

== 模型结构

#img(
  image("../opening/assets/paper-musegan.png", width: 85%),
  caption: [MuseGAN 论文中给出的整体结构示意 @dong2018musegan],
) <figure:paper-musegan>

#img(
  image("../methodology/assets/gan-outline.png", width: 75%),
  caption: [MuseGAN 的对抗模型]
) <figure:gan-outline>

如@figure:gan-outline，MuseGAN 的模型由两个主要部件构成：生成模块 $G$ 从随机噪声中生成乐谱数据，鉴别模块 $D$ 尝试将生成的乐谱数据与真实训练数据进行二分类 @dong2018musegan。对 $D$ 的训练能够促进 $G$ 产生更为符合真实乐谱性质的输出，从而起到对抗的效果。

=== 鉴别模块

#img(
  image("../methodology/assets/d-outline.png", width: 50%),
  caption: [鉴别模块的结构]
) <figure:d-outline>

如图@figure:d-outline，鉴别模块对于每个 $(G(z), x)$ 二元组计算三种损失：分类 $G(z)$ 的 Wasserstein 距离（即精确率的相反数）、分类 $x$ 的 Wasserstein 距离、分类 $0.5 G(z) + 0.5x$ 的梯度惩罚值。最后一项损失可以有效防止鉴别模块产生过拟合。图中 $C$ 单元结构如@figure:critic。

#img(
  image("../methodology/assets/critic.png"),
  caption: [$C$ 的结构]
) <figure:critic>

=== 生成模块

#img(
  image("../methodology/assets/g-outline.png", width: 75%),
  caption: [生成模块的结构]
) <figure:g-outline>

#img(
  image("../methodology/assets/temporal-net.png"),
  caption: [$G_#text([temp])$ 的结构]
) <figure:temporal-net>

#img(
  image("../methodology/assets/bar-gen.png"),
  caption: [$G_#text([bar])$ 的结构]
) <figure:bar-gen>

生成模块由多个子网络组成，共同完成从随机噪声到多轨音乐片段的生成。整体结构如@figure:g-outline 所示，生成器接收四种噪声输入：和弦（chords）、风格（style）、旋律（melody）和节奏（groove）。其中和弦和风格为全局共享，旋律和节奏则针对每个音轨独立生成。

和弦生成器和旋律生成器均为时态网络，其结构如@figure:temporal-net 所示。时态网络通过转置卷积将一维噪声向量扩展为二维特征图，最终输出形状为 $(B, Z, N_B)$ 的时态特征序列。每个音轨的时态特征与和弦、风格、节奏特征拼接后，送入对应的小节生成器。

小节生成器结构如@figure:bar-gen 所示，采用全连接层和转置卷积层的级联结构。输入的四维噪声向量首先通过全连接层映射到高维空间，再经过四次转置卷积操作逐步上采样，最终输出单条音乐数据。所有音轨和所有小节的输出拼接后，形成完整的 $(B, N_T, N_B, N_S, N_P)$ 五维张量。

== WGAN-GP

本实验对抗模型训练采用带梯度惩罚的 Wasserstein GAN（WGAN-GP） @gulrajani2017wgangp 模式。在每个 batch 中，先随机采样多个潜变量，以其为模板使用生成器产生负向样本，用负向样本训练判别器，由正样本损失、负样本损失、梯度惩罚加和得到总体的判别器损失，在判别器中进行反向传播；最后训练一次生成器，使生成样本的判别结果输出趋向正样本。整个过程的效果是，判别器变得更能区分训练数据中的正向样本与潜变量生成的负向样本，生成器能够从潜变量中生成更符合训练数据的输出。

#img(
  image("assets/wgan-gp.png"),
  caption: [WGAN-GP 算法]
)

WGAN-GP 算法的核心在于 Wasserstein 损失，实现为 $cal(L)(y, t) = -E[y dot t]$，本实验中使用 $t in {+1, -1}$ 来区分“希望分数更高/更低”的优化方向。对于每个混合样本 $tilde(x) = epsilon x + (1 - epsilon) hat(x)$，计算判别器输出对输入的梯度范数 $E[(1 - ||nabla_(tilde(x)) D(tilde(x))||_2)^2]$，乘上常数 $lambda = 10$ 后作为最终的梯度惩罚。

= 实验设置

== 实验环境

本实验环境如下：

- 软件：Ubuntu 22.04、Python 3.13、torch 2.9.0；
- 训练硬件：AutoDL 48GB vGPU (NVIDIA RTX 4090 48GB)；
- 推理硬件：NVIDIA RTX 4060 Laptop 8GB。

== 数据准备与加载

获取原始 LPD 数据集后，需要使用数据预处理脚本将输入数组转置为 $(N, N_T, N_B, N_S, N_P)$。原文中提供的脚本仅能在 Linux 环境下使用且性能不佳，本实验中使用的脚本对其进行了改进，能够跨平台、高性能进行预处理。

数据集类为 `LPDDataset`，读取与论文中同格式 npz 对象的 `arr_0` 并转为浮点张量；DataLoader 设置随机数种子以保证可复现性。

== 超参数与训练策略

训练脚本 3-musegan/3-musegan-train.py 固定的核心超参数如@table:hyperparam 所示：

#tbl(
  table(
    columns: 3,
    align: (center, center, center),
    toprule(),
    table.header([参数], [取值], [说明]),
    midrule(),
    [$N_B$], [4], [生成/训练的小节数],
    [$N_T$], [5], [轨道数],
    [$N_S$], [48], [每小节步数],
    [$N_P$], [84], [音高维度],
    [$Z$], [32], [噪声向量维度],
    [$B$], [64], [训练 batch 大小],
    [epochs], [25], [单次运行训练轮数],
    [repeat], [5], [Trainer 默认值，每 batch 中训练判别器时采样次数],
    [lr], [0.0001], [生成器与 critic 的 Adam 学习率],
    [$beta_1, beta_2$], [0.2, 0.9], [Adam 参数],
    [seed], [0x0D000721], [PyTorch/NumPy/`random` 统一随机种子],
    bottomrule(),
  ),
  caption: [训练脚本中的核心超参数],
) <table:hyperparam>

每个 epoch 保存一次训练检查点，存储，用于结果推理与评估。

= 实验结果与分析

== 推理生成流程与后处理

推理脚本首先初始化生成器与优化器，加载检查点的优化器状态，从检查点载入生成器权重。之后，采样四类噪声：

- `chords`：`torch.rand(B, Z)`；
- `style`：`torch.rand(B, Z)`；
- `melody`：`torch.rand(B, N_T, Z)`；
- `groove`：`torch.rand(B, N_T, Z)`。

将噪声潜变量前向通过生成器网络，经过后处理写入 MIDI 文件。

后处理使用 $arg max$ 将音高维度离散化为单音高序列，再以步长 0.25（四分之一拍长度）累积音符时值，最后并在“音高变化或每 4 步边界”处切分音符。为实现简单起见，本实验中的后处理方式将每个轨道当作单声部旋律线处理，并未设置 MIDI 程序或鼓组通道，与原 MuseGAN 包含鼓组等不同乐器语义的听感会有一定差异。

== 训练稳定性与损失曲线

#grid(
  columns: 3,
  gutter: 12pt,
  img(
    image("assets/real-fake-gap.png", width: 100%),
    caption: [正向样本与负向样本的判别器损失]
  ),
  img(
    image("assets/gen-loss.png", width: 100%),
    caption: [生成器损失随 epoch 变化曲线]
  ),
  img(
    image("assets/crit-loss.png", width: 100%),
    caption: [判别器损失随 epoch 变化曲线]
  ),
) 

如图所示，在 25 轮训练过程中，WGAN-GP 能够正确训练判别器，可由图中正负向样本损失递减但间距保持非零说明。生成器与判别器均能够在少数轮内达到较好的收敛效果。

#img(
  image("assets/midi.png"),
  caption: [推理产物的 MIDI 编辑器视图]
) <figure:midi>

根据试听与 MIDI 音轨查看器分析，25 轮训练模型的推理产物与原文在听感上有较大差距，在旋律走向、音色选择上仍非常不足，但是已经能看出较为清晰的旋律与乐句结构。

= 研究总结与展望

本次实验围绕 MuseGAN 的关键设计完成了端到端的复现与验证，包括多轨钢琴卷轴的张量化表示、生成器的层次化噪声注入与按小节生成的结构组织、以及基于三维卷积的判别器判别结构，并使用 WGAN-GP 的训练范式获得了可收敛的训练过程。配套实现覆盖了数据预处理、训练与断点续训、推理采样与 MIDI 导出，使得实验流程能够在给定随机种子与超参数配置下重复运行，便于后续对比不同结构或训练策略的效果。

从训练曲线与推理样例来看，WGAN-GP 的梯度惩罚对稳定训练确有帮助，判别器能够维持真实样本与生成样本评分差距，生成器也能在有限轮数内学到一定的节奏密度与短程重复模式，生成结果在 MIDI 编辑器中呈现出可辨识的片段结构。然而，与论文目标的“多轨协同编配”相比，当前产物仍存在明显不足：旋律线条的可听性与动机发展较弱，轨道之间的和声协同不稳定，且由于后处理采取了逐步 $arg max$ 的单音高离散策略并将每轨视作单声部，模型输出中潜在的复调信息难以被表达为自然的音符事件序列；同时，未对 MIDI 轨道设置明确的乐器音色与鼓组通道，也会使主观听感与论文设置存在偏差。

后续改进可以从“表示——解码——评估”三个层面同步推进。在表示与解码上，可将轨道语义与论文的五轨定义对齐，为鼓轨使用 drum channel 并设定程序号，从而提升多轨可解释性与试听一致性；同时引入更合理的事件化解码（例如基于阈值/采样的多音符激活、最小时值约束与连音合并策略），以避免 $arg max$ 带来的信息塌缩并提升乐句连贯性。在训练与模型上，可进一步做超参数与结构消融，如判别器训练重复次数、学习率、梯度惩罚权重、归一化与谱归一化等，以改善模式崩塌与轨道失衡问题，并尝试在更接近论文的数据划分与更长训练时长下比较差异。在评估方面，应使用如 EB、UPC、QN、DP 或 TD 等客观指标与训练过程联动记录，从而更清晰地定位性能瓶颈并指导迭代方向。

#pagebreak()

= 附录：实验源代码

本实验的源代码可在 #link("https://github.com/CSharperMantle/hdu2025_deep_learning/tree/main/3-musegan") 处获得。本实验使用如@code:training 所示的工作流训练。

#code(
  ```bash
  python scripts/prepare_data.py $path_to_raw_data --out-dir prepared

  python 3-musegan-train.py

  python 3-musegan-train.py --resume --from-epoch 38

  python 3-musegan-eval.py
  ```,
  caption: [训练与推理流程],
) <code:training>

#pagebreak()

#bibliography("bib.bib", style: "gb-7714-2015-numeric")
