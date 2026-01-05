# Day 1｜LLM 后训练与持续学习：课程总览

[TOC]

## 1. 为什么设计这门课程

在大模型时代，**预训练（Pre-training）** 只是起点。一个真正可用、可信、可控的 LLM，还需要经历一系列的**后训练（Post-training）** 流程：从监督微调（SFT）让模型学会"如何说话"，到偏好优化（DPO）和强化学习（RLHF）让模型学会"说什么更好"，再到持续学习（Continual Learning）让模型"与时俱进、学新不忘旧"。

本课程的设计遵循**由浅入深、由基础到前沿**的原则，将 LLM 后训练的完整流程拆解为六个阶段，每个阶段既独立成体系，又环环相扣：

```
Day 2 SFT 实战 → Day 3 偏好优化 → Day 4 RL理论 → Day 5 LLM-RL实践 → Day 6 CL基础 → Day 7 LLM-CL前沿
      ↓              ↓               ↓              ↓              ↓              ↓
  "学会说话"     "说得更好"      "理论武装"     "把事办成"      "防止遗忘"     "持续进化"
```

<br>

## 2. 课程内容总览

### 2.1 Day 2：SFT 监督微调实战

**核心目标**：让预训练模型从"续写文本"转变为"遵循指令"。

**主要内容**：
- **QLoRA 技术**：4-bit 量化 + LoRA 低秩适配，在单卡 4080 上训练 4B-8B 模型
- **Assistant-only Loss Mask**：只对 assistant 回复计算损失，避免模型"背诵"用户输入
- **数据集工程**：规范化、去重、混合采样、packing 等工业级数据处理流程
- **Qwen3 模型实战**：从加载到训练到评测的完整流程

**为什么先学 SFT**：SFT 是后训练的**基石**。它为后续的 DPO/RL 提供了一个"可用的起点"——一个已经懂得指令格式、能产出结构化回复的模型。没有 SFT，偏好优化和强化学习将无从谈起。

<br>

### 2.2 Day 3：偏好优化（DPO）

**核心目标**：让模型学会"什么回答更好"，而不仅仅是"模仿标准答案"。

**主要内容**：
- **偏好学习理论**：Bradley-Terry 模型、打分器视角与选择概率视角
- **DPO 公式推导**：从 RLHF 目标函数推导出直接偏好优化损失
- **DPO 实战**：数据集构建（chosen/rejected 对）、训练配置、β 参数调优
- **与 RLHF 的对比**：DPO 绕开了显式奖励模型，训练更简单稳定

**为什么在 SFT 之后学 DPO**：SFT 教模型"如何说话"，但无法教模型"什么话更好"。面对主观性问题（如"写一首赞美秋天的诗"），存在无数正确答案，SFT 只能模仿其一。DPO 通过成对偏好学习，让模型理解"更优"的概念，这是从"模仿者"到"优化者"的关键跨越。

<br>

### 2.3 Day 4：强化学习理论基础

**核心目标**：为 LLM-RL 打下理论基础，理解"为什么需要 RL"以及"RL 如何工作"。

**主要内容**：
- **MDP 建模**：状态、动作、奖励、策略、价值函数
- **Bellman 方程**：最优性原理、价值迭代、策略评估
- **Value-based RL**：Q-Learning、DQN、TD 误差
- **Policy-based RL**：策略梯度定理、REINFORCE、Actor-Critic
- **PPO 算法详解**：重要性采样、Clip 机制、KL 惩罚、GAE 优势估计

**为什么要学 RL 理论**：虽然 DPO 可以解决 80% 的对齐问题，但对于**结果导向、不可微分**的目标（如"代码是否通过单测"、"JSON 是否合法"、"是否正确调用工具"），RL 是唯一可行的优化框架。理解 RL 的数学原理，才能在实践中正确调参、诊断问题。

<br>

### 2.4 Day 5：LLM 中的强化学习实践

**核心目标**：将 RL 理论应用于 LLM，实现"让模型把事办成"。

**主要内容**：
- **LLM 的 MDP 建模**：状态=上下文、动作=下一个 token、奖励=序列级打分
- **奖励模型（RM）**：Bradley-Terry 训练、ORM vs PRM
- **PPO-RLHF**：完整流程、Rollout 采样、ptx 正则
- **GRPO**：组内相对优化、无 Critic 的优势估计
- **GSPO**：序列级重要性采样、长序列稳定性
- **DAPO**：Clip-Higher、动态采样、Token-level 损失、超长惩罚

**为什么专门讲 LLM-RL**：LLM 的 RL 与传统游戏/机器人 RL 有本质区别：动作空间巨大（词表数万）、奖励稀疏（序列结束才给分）、分布敏感（需要 KL 约束防止崩塌）。Day 5 聚焦这些 LLM 特有的挑战与解决方案。

<br>

### 2.5 Day 6：持续学习基础

**核心目标**：理解"如何让模型学新不忘旧"，掌握经典 CL 方法。

**主要内容**：
- **灾难性遗忘**：神经网络顺序学习新任务时丢失旧知识的本质原因
- **三大方法族**：
  - **Replay-based**：Experience Replay、iCaRL（近均值分类器 + 蒸馏 + Herding 选样）
  - **Regularization**：EWC（Fisher 信息矩阵约束）、LwF（知识蒸馏）、GEM（梯度投影）
  - **参数隔离**：PackNet（剪枝+掩码）、LoRA/Adapter、MoE/Lifelong-MoE
- **方法对比**：不同方法在数据需求、任务 ID 依赖、遗忘控制等维度的权衡

**为什么在 RL 之后学 CL**：无论是 SFT、DPO 还是 RL，都假设训练数据一次到位。但现实中，用户偏好在变、知识在更新、新工具在涌现。Day 6 引入"时间"维度，让模型具备**终身学习**的能力。

<br>

### 2.6 Day 7：LLM 与持续学习前沿

**核心目标**：将 CL 方法应用于 LLM 的全生命周期，了解最新研究进展。

**主要内容**：
- **LLM-CL 的三个阶段**：
  - **持续预训练（CPT）**：更新时效知识、扩展领域/语言
  - **持续微调（CFT/CIT）**：任务递增、领域递增、工具递增
  - **持续对齐（CA）**：偏好演化、安全策略更新
- **前沿算法**：
  - **CPPO**：样本分类 + 加权策略学习/知识保留
  - **COPR**：最优策略正则 + 适度奖励函数（MRF）
  - **InsCL**：基于指令语义的动态重放
  - **Multi-Stage Fine-Tuning**：偏好学习偏置 + 自蒸馏增强

**为什么以 LLM-CL 收尾**：这是将所有技术融会贯通的综合实践。一个工业级 LLM 系统，需要在 SFT/DPO/RL 的基础上，持续吸收新知识、适应新偏好、掌握新工具——Day 7 正是这一愿景的技术路线图。

<br>

## 3. 课程设计的逻辑链条

我们这样安排课程的原因，可以从三个维度理解：

### 3.1 技术依赖链

```
预训练模型 → [Day 2 SFT] → 指令模型 → [Day 3 DPO] → 偏好模型 → [Day 4-5 RL] → 对齐模型
                                                                    ↓
                                                           [Day 6-7 CL] → 持续进化模型
```

每个阶段的输出，都是下一阶段的输入。SFT 产出的模型作为 DPO 的参考策略（π_ref）；DPO/SFT 产出的模型作为 RL 的初始策略；所有训练好的模型都面临"如何持续更新"的 CL 问题。

### 3.2 难度递进

| 阶段 | 核心概念 | 数学难度 | 工程复杂度 |
|------|----------|----------|------------|
| Day 2 SFT | 交叉熵、LoRA | ⭐ | ⭐⭐ |
| Day 3 DPO | 偏好模型、KL 散度 | ⭐⭐ | ⭐⭐ |
| Day 4 RL 理论 | MDP、Bellman、策略梯度 | ⭐⭐⭐ | ⭐ |
| Day 5 LLM-RL | PPO/GRPO/DAPO | ⭐⭐⭐ | ⭐⭐⭐ |
| Day 6 CL 基础 | Fisher、蒸馏、MoE | ⭐⭐ | ⭐⭐ |
| Day 7 LLM-CL | 综合应用 | ⭐⭐⭐ | ⭐⭐⭐ |

我们把理论最重的 Day 4 放在中间，前有 SFT/DPO 的实战铺垫，后有 Day 5 的应用落地，避免"理论悬空"。

### 3.3 实用价值链

- **Day 2-3**：快速上手，80% 的场景用 SFT+DPO 就能解决
- **Day 4-5**：攻克难题，应对"不可微目标"和"结果导向"任务
- **Day 6-7**：持续运营，让模型在生产环境中长期保持竞争力

<br>

## 4. 贯穿全课程的核心主题

### 4.1 稳定性 vs 可塑性（Stability-Plasticity Dilemma）

这是持续学习的核心矛盾，但它实际上贯穿所有后训练阶段：

- **SFT**：学习新格式，可能遗忘预训练知识
- **DPO**：学习新偏好，可能丢失指令遵循能力
- **RL**：追逐高奖励，可能输出变得"面目全非"（需要 KL 约束）
- **CL**：学习新任务，如何不忘旧任务

**解决思路一致**：KL-to-reference（DPO/RL）、Replay（CL）、参数隔离（LoRA/MoE）。

### 4.2 KL 散度作为"锚点"

KL 散度在整个课程中反复出现：

- **DPO**：隐式 KL 约束嵌入损失函数
- **PPO**：KL 惩罚项或自适应 β
- **GRPO/GSPO**：对参考策略的 KL 正则
- **LwF/蒸馏**：输出分布的 KL 约束
- **COPR**：对历史最优策略的 KL 约束

理解 KL 散度，就理解了"如何让模型更新但不失控"的核心机制。

### 4.3 从"监督"到"偏好"到"奖励"

```
SFT：学习 (x, y)，最小化 -log p(y|x)
DPO：学习 (x, y⁺, y⁻)，让 p(y⁺) > p(y⁻)
RL：学习 (x, y, r)，最大化 E[r(x,y)]
```

信号越来越弱（从完整答案到偏好到标量分数），但优化目标越来越灵活。

<br>

## 5. 学习建议

1. **先跑通代码，再理解原理**：Day 2-3 的代码相对简单，建议先动手实践，再回头看理论。

2. **Day 4 是关键转折点**：RL 理论是最抽象的部分，建议反复阅读 Bellman 方程和策略梯度定理，直到真正理解。

3. **统一的评测面板**：从 Day 2 开始，就建立统一的评测指标（PPL、可用性、JSON 合法率、win-rate），便于跨阶段对比。

4. **参考策略的重要性**：Day 2 的 SFT 模型将作为后续所有阶段的"锚点"（π_ref），务必训练好这个基础。

5. **小模型先验证，大模型再部署**：课程使用 Qwen3-4B 等小模型，方便在单卡上快速迭代。

<br>

## 6. 课程产出物

完成本课程后，你将获得：

| 阶段 | 产出物 |
|------|--------|
| Day 2 | SFT LoRA 适配器 + 评测基线表 |
| Day 3 | DPO 训练代码 + 偏好数据处理流程 |
| Day 4 | Q-Learning 可视化 Demo |
| Day 5 | PPO/GRPO 训练脚本 + 奖励模型 |
| Day 6 | EWC/LwF/iCaRL 实现代码 |
| Day 7 | 完整的 LLM 持续学习 Pipeline |

<br>

***我们近期还会推出全流程后训练项目，使用后训练技术在多个领域来训练和评估一个基于MBTI人格分类的小模型（MBTI-Reasoning）。敬请期待！***

## References

[1] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *arXiv:2203.02155*.

[2] Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*.

[3] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

[4] Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.

[5] Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.

[6] Rebuffi, S., et al. (2017). iCaRL: Incremental Classifier and Representation Learning. *CVPR*.

[7] Li, Z., & Hoiem, D. (2017). Learning without Forgetting. *TPAMI*.

[8] Wu, T., et al. (2024). Continual Learning for Large Language Models: A Survey. *arXiv:2402.01364*.

[9] Shi, H., et al. (2024). Continual learning of large language models: A comprehensive survey. *ACM Computing Surveys*.

[10] Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv:2305.14314*.
