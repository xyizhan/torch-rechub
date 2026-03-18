---
title: 生成式推荐发展脉络与论文导读
description: 系统梳理生成式推荐的研究主题、方法演进、代表论文与学习路径，并逐项核对论文题目和链接。
---

# 生成式推荐发展脉络与论文导读

这篇笔记面向想系统学习“生成式推荐”的读者，目标不是罗列名词，而是回答三个更关键的问题：

1. 什么叫“生成式推荐”，它和传统检索式推荐到底差在哪里？
2. 这个方向是如何从“把推荐改写成文本任务”，一步步发展到“生成语义 ID”“生成篮子”“用 LLM 做排序/对齐”“工业级生成式序列建模”的？
3. 每一类方法的核心原理、优势、局限与代表论文分别是什么？

本文只收录我逐项核对过标题与链接对应关系的论文。开放论文优先使用 arXiv / PMLR / DOI；个别 ACM 论文在公开搜索不便时，使用 DOI 并辅以会议 accepted page 做交叉核对。

---

## 1. 先给一个总图：什么是生成式推荐

传统推荐系统大多是“从候选集里找东西”：

- 召回阶段先从海量已有物品中检索候选；
- 排序阶段再估计用户对候选物品的偏好分数；
- 最终输出的是“从现有 item 库中挑选出来的结果”。

生成式推荐则把问题往前推了一步。它的“生成”大致有四种层次：

| 层次 | 生成的对象 | 代表问题 |
| --- | --- | --- |
| 文本统一建模 | 把推荐任务统一转写为文本序列 | 能否把多种推荐任务放进同一个 seq2seq 框架？ |
| ID / 语义 token 生成 | 直接自回归生成 item ID 或 semantic ID | 能否不用 ANN 检索，直接 decode 出目标 item？ |
| 集合 / 表示生成 | 生成下一篮子物品、生成目标 item 表示分布 | 能否显式建模 item-item 关系与不确定性？ |
| 内容与指令生成 | 根据自然语言意图生成推荐结果或推荐内容 | 能否让用户通过指令参与推荐，并突破固定物品库？ |

所以“生成式推荐”不是单一技术，而是一组范式变化。它的主线可以概括成一句话：

**从“给定用户，给候选 item 打分”，转向“给定上下文，直接生成推荐输出”。**

---

## 2. 发展时间线

| 时间 | 代表论文 | 它推动了什么变化 |
| --- | --- | --- |
| 2022 | [P5](https://arxiv.org/abs/2203.13366), [M6-Rec](https://arxiv.org/abs/2205.08084) | 把推荐统一成语言建模 / 基础模型问题 |
| 2023 上半年 | [GeneRec](https://arxiv.org/abs/2304.03516), [DiffuRec](https://arxiv.org/abs/2304.00686), [TALLRec](https://arxiv.org/abs/2305.00447), [InstructRec](https://arxiv.org/abs/2305.07001), [LLMRank](https://arxiv.org/abs/2305.08845) | 形成“生成式推荐”概念，开始探索扩散、指令微调、零样本排序 |
| 2023 下半年 | [TIGER / Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065), [GeRec](https://doi.org/10.1145/3604915.3608823) | 生成式召回和生成式篮子推荐成为可落地路线 |
| 2024 | [HSTU](https://proceedings.mlr.press/v235/zhai24a.html), [Recommender Systems in the Era of Large Language Models (LLMs)](https://arxiv.org/abs/2307.02046) | 生成式推荐从研究原型走向工业规模与系统化总结 |

如果只看主干演化，大致可以分成 4 条线：

1. 把推荐转成语言任务；
2. 让模型直接生成 item token / semantic ID；
3. 让模型生成篮子或分布式表示；
4. 用 LLM 的提示、指令微调和推理能力改造推荐系统。

---

## 3. 主题一：把推荐统一成语言建模任务

这一阶段的关键想法是：推荐系统长期被拆成很多相互独立的任务，导致知识很难共享。如果把用户行为、物品属性、评论、任务目标都改写成自然语言序列，那么一个统一的 seq2seq 模型就有机会跨任务迁移。

### 3.1 P5：Recommendation as Language Processing

论文：[Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)](https://arxiv.org/abs/2203.13366)

核心思想：

- 将用户交互、物品元数据、评论等全部转换成自然语言序列；
- 用统一的语言建模目标训练同一个模型；
- 通过 personalized prompt 把不同推荐任务映射到统一接口。

为什么重要：

- 它不是简单“拿语言模型做推荐”，而是较早系统提出“推荐即语言处理”的统一范式；
- 后续很多 LLM4Rec 工作，本质上都在延续 P5 的接口思想，只是底座模型和训练策略更强了。

局限：

- 主要是“把已有推荐任务翻译成文本任务”，并没有真正解决海量 item 空间下的高效生成问题；
- 对推荐领域的结构化信号利用仍然受限于文本化表达。

### 3.2 M6-Rec：开放域、开放任务的基础模型尝试

论文：[M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems](https://arxiv.org/abs/2205.08084)

核心思想：

- 把工业推荐系统看成多域、多任务、开放场景的统一建模问题；
- 用 generative pretrained language model 支撑“开放式推荐系统”。

它相对 P5 的推进：

- P5 更强调“统一任务接口”；
- M6-Rec 更强调“推荐基础模型”与开放域任务扩展性。

这一阶段的方法论结论：

- 推荐系统可以被重写成序列生成问题；
- 统一预训练是可行的；
- 但如果要真正触及工业召回与大规模 serving，仍然需要新的 item 表示与解码机制。

---

## 4. 主题二：从“统一文本任务”走向“真正的生成式推荐范式”

2023 年开始，研究者开始明确意识到：如果推荐仍然只能从固定 item 库中检索，那么它只是“生成式建模的外壳”，还不算彻底改变推荐范式。

### 4.1 GeneRec：明确提出“下一代推荐范式”

论文：[Generative Recommendation: Towards Next-generation Recommender Paradigm](https://arxiv.org/abs/2304.03516)

这篇论文的价值，主要不在于某个具体 SOTA 模型，而在于它把“生成式推荐”这个概念讲清楚了。

作者认为传统 retrieval-based recommender 有两个天然限制：

- 只能在现有 item 库里找结果，无法生成真正匹配用户需要的新内容；
- 用户对推荐结果的控制主要依赖被动反馈，交互效率低。

因此它提出 GeneRec 这个更大的范式设想：

- 引入用户 instruction；
- 让系统不只是“检索 item”，而是可以编辑、重组、乃至创造内容。

这篇论文更像方向宣言：

- 它把“生成式推荐”从模型技巧上升为推荐系统目标函数的改变；
- 后续的 instruction-based rec、AIGC rec、可控推荐生成，很多都能在这里找到思想源头。

---

## 5. 主题三：生成 item，而不是检索 item

这是生成式推荐最有辨识度的一条主线。核心问题是：

**如果推荐目标 item 不再通过 ANN 检索得到，而是由模型像语言模型生成 token 一样一步步 decode 出来，会发生什么？**

### 5.1 TIGER：语义 ID + 自回归解码

论文：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)

这篇论文里最有影响力的对象是 TIGER 范式。它的关键机制是：

1. 不再直接使用离散、稀疏、语义贫乏的原始 item ID；
2. 先为每个 item 构造由多个 codeword 组成的 **Semantic ID**；
3. 再把“预测下一个 item”改写成“自回归生成下一个 item 的 semantic ID 序列”。

为什么这一步很关键：

- 它让生成式召回开始具备较清晰的工程形态；
- semantic ID 让相似 item 在 token 空间共享结构，改善冷启动和泛化；
- 召回问题被更自然地改写成 seq2seq 解码问题。

方法本质：

- 传统双塔召回：`user embedding -> ANN -> item`
- TIGER：`user history -> decoder -> semantic tokens -> item`

挑战：

- semantic ID 的构造质量直接决定上限；
- 解码错误会级联；
- 真正工业落地时仍要平衡延迟、词表设计和训练稳定性。

### 5.2 为什么 TIGER 是分水岭

P5 / M6-Rec 更多是在“统一接口”层面改造推荐；TIGER 则开始碰推荐系统最核心的检索机制本身。

你可以把这个变化理解成：

- 早期工作在问：“能不能把推荐写成文本任务？”
- TIGER 在问：“能不能把 item 检索本身变成 token 生成？”

这也是为什么后续工业界会非常关注 semantic token、RQ-VAE、generative retrieval 这条线。

---

## 6. 主题四：生成集合与生成分布

并不是所有推荐输出都适合“生成单个 item ID”。有些任务天然是集合生成或者多模态不确定性建模，于是出现了另外两条重要分支。

### 6.1 GeRec：生成下一篮子，而不是独立打分每个 item

论文：[Generative Next-Basket Recommendation](https://doi.org/10.1145/3604915.3608823)

任务背景：

- 在 next-basket recommendation 里，目标不是“下一个 item”，而是“下一个 basket”；
- 如果仍然把每个 item 独立打分，容易忽视 basket 内 item-item 关系，结果会过于同质化。

GeRec 的核心思想：

- 用自回归 decoder 逐个生成下一篮子的 item；
- 同时建模用户的 item-level 与 basket-level 上下文；
- 让“篮子内部物品之间的关系”直接体现在生成过程中。

为什么重要：

- 它说明生成式推荐并不只适用于单 item 召回；
- 对于套餐、购物篮、组合推荐这类任务，生成式建模尤其自然。

### 6.2 DiffuRec：把目标 item 表示看成分布，而不是固定向量

论文：[DiffuRec: A Diffusion Model for Sequential Recommendation](https://arxiv.org/abs/2304.00686)

DiffuRec 的关键转变是：

- 传统序列推荐把 item 表示当成固定点向量；
- DiffuRec 认为用户兴趣和 item 语义本身存在不确定性与多样性，因此更适合生成“分布式表示”。

原理上，它把扩散模型引入 sequential recommendation：

- 正向过程对目标 item embedding 加噪；
- 反向过程在用户历史条件下逐步去噪，重建目标表示；
- 再将重建后的表示映射回目标 item。

这条线的价值：

- 它扩展了“生成”的含义，不只是生成离散 token，也可以生成连续表示；
- 很适合解释多兴趣用户、语义不确定性、长尾 item 的表达问题。

局限：

- 推理成本通常高于简单判别式模型；
- 生成的连续表示最终仍需映射回离散 item，服务链路更复杂。

---

## 7. 主题五：LLM 进入推荐系统后的三种典型角色

2023 年之后，生成式推荐又出现了一条非常强的支线：不是直接生成 item ID，而是把 LLM 当成推荐系统中的新模块。大致有三种角色。

### 7.1 角色 A：通过微调把 LLM 对齐到推荐任务

代表论文：[TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation](https://arxiv.org/abs/2305.00447)

核心问题：

- 通用 LLM 有语言能力，但没有足够强的 recommendation inductive bias；
- 直接拿来做推荐，效果通常不稳定。

TALLRec 的做法：

- 不把 LLM 神化为“开箱即用推荐器”；
- 而是通过高效 tuning，把推荐数据注入到 LLM 中；
- 让模型更接近“Recommendation LLM”。

方法意义：

- 它把“LLM 能不能做推荐”的讨论，从 prompt-only 推进到 alignment / fine-tuning；
- 说明 LLM4Rec 的关键不是只会写 prompt，而是要处理任务不匹配问题。

### 7.2 角色 B：把推荐显式写成 instruction following

代表论文：[Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach](https://arxiv.org/abs/2305.07001)

这篇论文通常被简称为 InstructRec 路线。它进一步强调：

- 用户偏好、意图、上下文、任务形式都可以写成 instruction；
- 通过 instruction tuning，让 LLM 更自然地承接推荐任务。

它和 P5 的区别在于：

- P5 是“文本统一接口”；
- InstructRec 是“指令对齐接口”；
- 前者更像统一预训练范式，后者更像 LLM 时代的推荐任务组织方式。

### 7.3 角色 C：让 LLM 直接做重排器

代表论文：[Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845)

这篇论文非常有代表性，因为它提醒我们：

- LLM 未必要替代整个推荐系统；
- 它也可以只接管“理解用户历史 + 重排候选”的最后一步。

核心结论：

- LLM 具备一定 zero-shot ranking 能力；
- 但会受历史顺序感知、prompt 位置偏差、流行度偏差影响；
- 因此更适合作为 candidate reranker，而不是单独承担全链路召回。

这条路线后来影响很大，因为工业系统更容易接受“LLM 做重排/解释/对话接口”，而不是直接替换底层召回系统。

---

## 8. 主题六：工业级生成式推荐与大规模序列建模

前面的工作更多在证明“生成式推荐可行”。真正让人看到工业规模潜力的，是大模型时代的大规模序列转导路线。

### 8.1 HSTU：把推荐重写成大规模生成式序列转导

论文：[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://proceedings.mlr.press/v235/zhai24a.html)

这篇论文的关键信号非常强：

- 它不再把生成式推荐当作一个小众实验方向；
- 而是明确提出“Generative Recommenders”并给出工业级架构 HSTU；
- 还报告了 1.5T 参数级别部署经验。

核心思想：

- 将推荐问题改写为 sequential transduction；
- 用 HSTU 架构建模高基数、非平稳、流式行为序列；
- 在长序列、超大规模训练下追求生成式推荐的精度与效率平衡。

为什么重要：

- 这说明生成式推荐不只是一组学术 demo，而是可能成为下一代大规模推荐基础设施；
- 研究重点开始从“能不能生成”转向“怎么在万亿参数和超长序列上高效生成”。

---

## 9. 综述视角：这个方向现在到底形成了什么知识框架

代表综述：[Recommender Systems in the Era of Large Language Models (LLMs)](https://arxiv.org/abs/2307.02046)

这篇综述不是“生成式推荐”的唯一总结，但它很适合作为中层地图。它把 LLM4Rec 大致分成：

- Pre-training
- Fine-tuning
- Prompting

如果把本文的脉络和这篇综述拼起来，可以得到更完整的结构：

| 研究主线 | 核心问题 | 代表论文 |
| --- | --- | --- |
| 统一文本范式 | 如何把多任务推荐统一成生成问题？ | P5, M6-Rec |
| 范式升级 | 为什么推荐系统需要从检索走向生成？ | GeneRec |
| 生成式召回 | 如何直接生成 item 的 token / semantic ID？ | TIGER |
| 集合 / 分布生成 | 如何生成篮子、表示分布与不确定性？ | GeRec, DiffuRec |
| LLM 对齐与使用 | LLM 在推荐中应该扮演什么角色？ | TALLRec, InstructRec, LLMRank |
| 工业规模架构 | 如何让生成式推荐跑到超长序列和大规模线上系统？ | HSTU |

---

## 10. 方法变迁背后的原理变化

如果你想真正理解“方法为什么会变”，最值得抓住的是这 5 个原理层面的变化。

### 10.1 从判别式打分到生成式建模

传统推荐多学习 `f(user, item)` 分数；生成式推荐更常学习：

- `p(item | history)`；
- `p(semantic_id | history)`；
- `p(basket | history)`；
- `p(instruction-conditioned output | context)`。

也就是从“比较候选”转向“建模条件分布”。

### 10.2 从原始 ID 到语义化 token

原始 item ID 稀疏、无语义、共享能力差。Semantic ID / tokenization 的核心收益是：

- 让相似 item 共享子结构；
- 缩小输出空间的学习难度；
- 让冷启动和泛化更有机会受益。

### 10.3 从单点预测到序列解码

生成式推荐天然更适合：

- next-item；
- next-basket；
- session continuation；
- 多步未来行为建模。

因为这些任务本来就具有序列依赖，而自回归解码能显式表达“前一步影响后一步”。

### 10.4 从静态表示到概率生成

DiffuRec 这类工作说明：推荐目标未必要是一个固定 embedding 点，可以是一个条件生成出来的表示分布。这对多兴趣、多解空间问题更自然。

### 10.5 从封闭反馈到可控交互

GeneRec、InstructRec、LLMRank 这条线说明，生成式推荐不只是模型结构变化，也是在改变人与推荐系统的接口：

- 用户不再只能 click / skip；
- 用户可以通过 prompt / instruction 明确表达需求；
- 推荐系统开始具备“可解释、可编辑、可交互”的生成能力。

---

## 11. 该怎么学：一个尽量不绕路的阅读顺序

如果你是第一次系统学习，我建议按下面顺序读：

1. [P5](https://arxiv.org/abs/2203.13366)
   先理解“推荐为什么可以被统一成语言任务”。
2. [M6-Rec](https://arxiv.org/abs/2205.08084)
   体会“推荐基础模型”与开放域任务的设想。
3. [GeneRec](https://arxiv.org/abs/2304.03516)
   理解“生成式推荐”为什么不是简单换个模型，而是目标范式的升级。
4. [TIGER](https://arxiv.org/abs/2305.05065)
   学最关键的技术拐点：semantic ID + generative retrieval。
5. [GeRec](https://doi.org/10.1145/3604915.3608823) 和 [DiffuRec](https://arxiv.org/abs/2304.00686)
   看生成式方法如何分别处理“集合生成”和“分布生成”。
6. [TALLRec](https://arxiv.org/abs/2305.00447), [InstructRec](https://arxiv.org/abs/2305.07001), [LLMRank](https://arxiv.org/abs/2305.08845)
   理解 LLM 在推荐系统里究竟应该放在哪里。
7. [HSTU](https://proceedings.mlr.press/v235/zhai24a.html)
   最后看工业规模生成式推荐的架构思路。
8. [LLM 时代推荐综述](https://arxiv.org/abs/2307.02046)
   回头建立自己的整体知识图谱。

---

## 12. 与本仓库的对应关系

如果你希望一边读论文，一边看这个仓库里的落地实现，可以从这些入口继续：

- HSTU 复现说明：[/zh/blog/hstu_reproduction](/zh/blog/hstu_reproduction)
- HLLM 复现说明：[/zh/blog/hllm_reproduction](/zh/blog/hllm_reproduction)
- HSTU 示例：`examples/generative/run_hstu_movielens.py`
- HLLM 示例：`examples/generative/run_hllm_movielens.py`
- TIGER 示例：`examples/generative/run_tiger_amazon_books.py`
- RQ-VAE 示例：`examples/generative/run_rqvae_amazon_books.py`

这也说明一个很重要的现实判断：

- 生成式推荐不是“只有 LLM”；
- 在工程上，它往往由 token 化、序列建模、检索重构、LLM 对齐、多阶段 serving 共同组成。

---

## 13. 已核验论文清单

下表只保留我逐项核验过题目与链接的论文。

| 论文 | 年份 | 主题 | 已核验链接 |
| --- | --- | --- | --- |
| Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5) | 2022 | 统一文本范式 | https://arxiv.org/abs/2203.13366 |
| M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems | 2022 | 推荐基础模型 | https://arxiv.org/abs/2205.08084 |
| Generative Recommendation: Towards Next-generation Recommender Paradigm | 2023 | 范式定义 | https://arxiv.org/abs/2304.03516 |
| DiffuRec: A Diffusion Model for Sequential Recommendation | 2023 | 扩散式序列推荐 | https://arxiv.org/abs/2304.00686 |
| TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation | 2023 | LLM 对齐 | https://arxiv.org/abs/2305.00447 |
| Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach | 2023 | 指令式推荐 | https://arxiv.org/abs/2305.07001 |
| Large Language Models are Zero-Shot Rankers for Recommender Systems | 2023 | LLM 重排 | https://arxiv.org/abs/2305.08845 |
| Recommender Systems with Generative Retrieval | 2023 | 生成式召回 / TIGER | https://arxiv.org/abs/2305.05065 |
| Generative Next-Basket Recommendation | 2023 | 篮子生成 | https://doi.org/10.1145/3604915.3608823 |
| Recommender Systems in the Era of Large Language Models (LLMs) | 2024 | 综述 | https://arxiv.org/abs/2307.02046 |
| Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations | 2024 | 工业级生成式推荐 | https://proceedings.mlr.press/v235/zhai24a.html |

核验说明：

- 上表中的 arXiv 链接，均检查过 arXiv `abs` 页标题与论文题目一致；
- HSTU 使用 PMLR 官方论文页核对标题与 venue；
- GeRec 使用 DOI `10.1145/3604915.3608823`，并与 RecSys 2023 accepted contributions 页面中的题目和作者信息交叉核对。

---

## 14. 最后给一个判断

如果只用一句话概括这个领域的演进，我会这样说：

**生成式推荐的发展，不是“把大语言模型硬塞进推荐系统”，而是推荐系统逐步学会用生成式建模来统一任务接口、重构 item 表示、直接解码目标、吸收自然语言指令，并最终走向大规模序列转导基础设施。**

真正值得长期跟踪的，不只是某一篇爆款论文，而是下面这三个长期问题：

- semantic ID / tokenization 会不会成为大规模推荐的新基础接口？
- LLM 在推荐里更适合做全链路生成器，还是做重排器、解释器、交互器？
- 生成式推荐如何在工业系统里同时满足效果、延迟、可控性和稳定性？

如果这三个问题你能持续带着去读论文，这条学习线就会越来越清晰。
