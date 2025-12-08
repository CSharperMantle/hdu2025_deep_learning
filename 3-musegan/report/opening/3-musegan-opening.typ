#import "@preview/ouset:0.2.0": underset
#import "@preview/booktabs:0.0.4": *

#import "../../../template/hdu-report-typst/template/template.typ": *

#show: booktabs-default-table-style
#show: project.with(
  title: "《深度学习》期末作业",
  subtitle: [选题报告：基于 MuseGAN 的 MIDI 片段生成],
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

= 研究背景

生成对抗网络在文本@xu2017attngan、图像@ramesh2021zeroshot、视频@chen2020videogan 生成等领域已有，但在符号化音乐生成方面仍然面临诸多挑战。与图像生成不同，音乐生成具有其独特的复杂性。其一为时间性；“音乐是时间的艺术。”@dong2018musegan 具有如乐句、小节、节拍等层次结构，需要有效的模型来处理时间依赖关系。其二为多轨依赖性；非平凡的音乐通常由多个乐器轨道组成，如交响乐中的不同声部或乐器。这些轨道在时间上不仅各自发展，而且存在紧密的互动和依赖，构成音乐的整体。其三为复调特性；音乐中的音符通常组合成和弦或旋律，在线性时间轴上加入了第二个维度，使得在自然语言处理中常用的按时间顺序引入音符的方法并不完全适合。

在 MuseGAN @dong2018musegan 之前的工作大多对这些问题进行了简化，例如 MidiNet@yang2017midinet 只生成单轨单声部旋律，或者将多声部音乐视为多个单声部旋律的简单组合，忽略了复杂的和声与对位关系。MuseGAN 提出了一种基于生成式对抗网络的新型框架，旨在生成具有和声及节奏结构的多轨复调音乐。

= 研究意义

MuseGAN 将卷积神经网络应用于音乐这种时序数据，通过处理“钢琴卷轴”表示法，探索了 GAN 在非图像领域的生成能力。这对于理解如何利用深度学习处理复杂的时间依赖和多维数据结构具有理论意义。

MuseGAN 研究的核心意义在于攻克多乐器协同生成的难点。论文中提出的“混合模型”结合了独立生成与统一生成的思想，利用共享的潜在向量来协调不同轨道的生成，从而模拟人类作曲家在编曲时对和声与整体结构的把控，对于提升生成音乐的听感和谐度有重要意义。

除了从头开始生成音乐，MuseGAN 还支持“音轨条件生成”，即给定一个已有的音轨（如主旋律），模型能够自动生成其余的伴奏轨道。这种功能在智能辅助作曲、自动编曲软件等领域具有极高的实际应用价值。

原文中，作者还提出了多种用于量化生成音乐片段质量的指标，如空小节（empty bars，EB）、音名丰度（used pitch classes per bar，UPC）、有效音符（ratio of qualified notes，QN，即长于 32 分的音符占比）、鼓谱（drum pattern，DP，8 或 16 拍子音符模式的占比）、调性距离（tonal distance，TD）@harte2006tonaldistance 等。这些指标在研究模型的内部结构、训练模型、筛选模型产物中起到了关键作用，也为之后的音频生成研究提供了参考。

= 研究内容

#img(
  image("assets/paper-musegan.png"),
  caption: [MuseGAN 模型结构@dong2018musegan]
) <figure:paper-musegan>

本次作业的目标是复现 MuseGAN 论文中的核心模型，并利用 Lakh Pianoroll Dataset@raffel2016million 进行训练和评估。原文重点使用经过清洗的 LPD-5-cleansed 子集，将音轨整合为贝斯、鼓、吉他、钢琴和弦乐五个核心轨道，以解决数据稀疏和不平衡问题。原文将 MIDI 文件转换为多轨钢琴卷轴张量格式，之后通过微调设置音符时值，以适应三连音和十六分音符等常见节奏模式。

如@figure:paper-musegan，原文提出的“混合模型”架构旨在解决多轨生成的依赖性问题。其生成器基于 CNN 构建，通过同时输入跨轨道共享的随机向量与各轨道私有的随机向量，用协调全局和声，同时保持各乐器独立特性，借此实现多乐器的协同生成。此外，为了赋予生成音乐时间上的连贯性，原文实现了分层时序模型，包含“时序结构生成器”与“小节生成器”，使模型能够以乐句为单位生成具有长程结构的音乐片段，而非仅仅生成孤立的小节。在训练机制与评估环节，原文采用带有梯度惩罚的 Wasserstein GAN 作为损失函数，以提升训练过程的稳定性并避免生成对抗网络常见的模式崩溃问题。

= 可行性分析与技术难点

== 模型实现

原文提供的模型代码使用 TensorFlow@abadi2015tensorflow 1.10 实现，在现代环境中较难部署。Kanametov@kanametov2021musegan 提供了使用 PyTorch 1@paszke2019pytorch 实现的版本，可供复现时参考。本实验的编程目标是使用 PyTorch 2@ansel2024pytorch 实现 MuseGAN 的模型结构与训练。

== 数据集大小与计算强度

本项目的可行性首先建立在充足且规模适度的数据集基础之上。复现工作将依托于原文作者筛选、清洗后的 LPD-5-cleansed 子集，其数据规模既保证了深度卷积神经网络有足够的样本进行特征学习，又处于个人计算机能够处理的范围内，不需要大规模存储集群即可完成数据的加载与迭代。

在计算强度与硬件需求方面，本文的复现同样具有较高的可行性。原文中指出，使用单块 Tesla K40m GPU 进行训练，每个模型的训练时长不超过 24 小时。Tesla K40m 是较早期的 GPU 架构，而目前个人电脑入门级配置包含的 NVIDIA RTX 4060 显卡在单精度浮点运算性能上为 Tesla K40m 的 2.5 倍@nvidia2013tesla。尽管 WGAN-GP 算法要求判别器的更新频率高于生成器，增加了一定的计算开销，但鉴于模型整体参数量和训练数据的规模，预计在现有的实验硬件条件下，可以在数小时内完成模型的收敛。因此，从算力资源的角度评估，本项目能够可以在期末大作业规定的时间窗口内完成训练与调试。

#pagebreak()

#bibliography("bib.bib", style: "gb-7714-2015-numeric")
