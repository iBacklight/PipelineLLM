# 2｜系统2风格与测试时算力（TTS）：概念、机制及效果

[TOC]



## 1. 系统1 vs 系统2推理风格与测试时算力（TTS）

### 1) 系统1与系统2推理风格

“系统1”与“系统2”是心理学中“双过程”理论的概念，用来描述人类两种不同的思维模式。

- **系统1**指快速、直觉化、自动化的决策过程，往往依赖启发式和模式匹配，速度快但可能不够严谨。
- **系统2**是缓慢、刻意、逻辑推理驱动的过程，更倾向于逐步分析和深思熟虑，能提高判断准确性并减少偏误。

在LLM中，可以类比地将传统的直接回答视为系统1风格——模型基于训练中学到的模式快速给出反应；而引入显式推理步骤的链式思考（Chain-of-Thought, CoT）等策略，则让模型表现出更接近系统2的行为。一般的基础LLM擅长系统1式的快速生成，但在需要复杂推理的问题上往往力不从心，因为它们尚未充分采用系统2那样逐步分析的思维过程[arxiv.org](https://arxiv.org/abs/2502.17419#:~:text=reasoning,like cognitive abilities)。

### 2) 测试时算力(TTS) 

***测试时算力*（Test-Time Compute/Test-Time Search）**指的是在模型推理阶段（而非训练阶段）投入额外的计算资源和步骤，让模型“花更多时间思考”以求解问题[venturebeat.com](https://venturebeat.com/ai/alibaba-researchers-unveil-marco-o1-an-llm-with-advanced-reasoning-capabilities#:~:text=OpenAI o1 uses “inference,in tasks with standard answers)。

通俗来说，就是模型在回答问题时并非一次性给出结果，而是允许多次生成、中间检查或搜索，从而利用额外算力换取更好的答案。这一思想也被称为**推理时扩展（inference-time scaling）**。OpenAI的o1/o3模型便运用了这一理念，在推理时执行更多计算循环、生成更多令牌并自我审阅，从而提高复杂推理任务的表现。这与仅靠增加模型参数或训练数据不同：TTS是在**不改变模型参数**的情况下，通过增加推理过程的深度或广度来提升性能。例如，o1模型的性能随着“思考时间”（推理算力）的增加而持续提高。简而言之，TTS让模型在回答时有机会进行更充分的搜索和思考。

### 3) 与传统CoT提示的区别

早期的链式思考(CoT)主要通过**提示工程**来诱导模型进行多步推理，例如在 Zero-shot-CoT 中附加“让我们一步步推理”等提示语，或在 Few-shot-CoT 中给出含解析步骤的示例。这些方法本质上仍是一次前向推理：模型依据提示在单轮生成中产出生硬的思维链。然而**传统CoT提示并不改变模型推理的机制**，它只是引导模型显性地吐出推理过程，但过程本身仍是贪心地逐词生成，缺乏全局搜索和验证[arxiv.org](https://arxiv.org/pdf/2305.10601#:~:text=Language models are increasingly being,that serve as intermediate)[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=* Test,systematic optimization at test time)。相比之下，系统2风格的TTS策略在推理阶段引入了算法上的改进：例如**多次采样和搜索、分支探索、结果汇总投票、动态中间检查和校正**等。这意味着模型可以在**一次询问中进行多轮推理交互或多条思路探索**，而不局限于固定的单通路输出。简言之，传统CoT属于*提示层面的技巧*，让模型按人类可读方式思考但**不具备试错和搜索能力**；而系统2 + TTS策略属于*推理过程的改造*，赋予模型在回答时进行更深入探索与校正的能力。这也带来了弹性的**算力-精度权衡**：我们可以通过投入更多推理算力让模型表现更佳，是以前纯提示工程所无法实现的[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=This scalability is a significant,compute resources at test time)。

<br>

## 2. 技术机制与例子：系统2/TTS的关键机制

现代的大模型推理策略中，一系列代表性方法（如OpenAI的 **o1**[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=Our large,are continuing to investigate them)、DeepSeek的 **R1**[arxiv.org](https://arxiv.org/abs/2502.17419#:~:text=Foundational Large Language Models ,like cognitive abilities)、**Self-Consistency**自洽解答[medium.com](https://medium.com/@johannes.koeppern/self-consistency-with-chain-of-thought-cot-sc-2f7a1ea9f941#:~:text=ADVANTAGEDESCRIPTIONQUANTIFICATIONImproved PerformanceSelf,more comprehensive understanding of complex)、**Tree-of-Thoughts**思维树搜索[arxiv.org](https://arxiv.org/pdf/2305.10601#:~:text=non,Code repo with)等）体现了系统2风格和测试时算力的核心机制。下面结合这些方法，介绍几项关键技术要点，并辅以实例说明：

- **长链条思考（逐步细化问题）**：系统2推理的基础是让模型生成**更长、更详细的思维链**，将复杂问题分解为一系列易于处理的步骤。这通常通过强化学习或有监督微调来训练模型学会逐步推理。例如，OpenAI的o1模型经过**大型强化学习训练**，掌握了在回答困难问题前进行长链思考的策略——它会识别并纠正自己的错误，将棘手步骤拆解得更细，并在当前思路行不通时尝试不同的方法[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=Similar to how a human,on several difficult problems below)。这种**分解与反思**能力极大提升了模型的推理表现[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=Similar to how a human,on several difficult problems below)。IBM对推理模型的分析也指出，不同于旧式模型直接给出答案，o1这类“推理模型”会**显式地逐步解析问题**，因此虽然回答慢一些，但能够解决更复杂的任务[ibm.com](https://www.ibm.com/think/news/deepseek-r1-ai#:~:text=Reasoning models became the hot,a “chain of thought” manner)。由此可见，**长链式的逐步推理使模型更接近人类的系统2思维模式**，奠定了其他高级机制的基础。

- **多样采样与投票（自洽式思考）**：单一路径的推理可能受到偶然生成误差的影响。**自洽式Chain-of-Thought (Self-Consistency)** 方法通过在同一问题上**采样生成多条不同的思维链**，从而探索多种解题思路[medium.com](https://medium.com/@johannes.koeppern/self-consistency-with-chain-of-thought-cot-sc-2f7a1ea9f941#:~:text=Mitigating these problems with our,technique results in these advantages)。然后，汇总这些思路的最后答案，采用投票或共识机制选出最可能正确的答案[medium.com](https://medium.com/@johannes.koeppern/self-consistency-with-chain-of-thought-cot-sc-2f7a1ea9f941#:~:text=ADVANTAGEDESCRIPTIONQUANTIFICATIONImproved PerformanceSelf,more comprehensive understanding of complex)。这一简单的集成策略能显著缓解单次生成的随机性和贪心偏差，对抗局部最优。例如，

  - 研究表明在数学和常识问答数据集上，引入自洽采样可以将模型准确率提升约 **3.9% 至 17.9%**[medium.com](https://medium.com/@johannes.koeppern/self-consistency-with-chain-of-thought-cot-sc-2f7a1ea9f941#:~:text=ADVANTAGEDESCRIPTIONQUANTIFICATIONImproved PerformanceSelf,more comprehensive understanding of complex)。
  - OpenAI o1模型也应用了类似思想：在2024年美国高中数学竞赛AIME中，GPT-4o模型一次性作答仅得出约12%的正确率，而o1模型单次作答能解出74%的题目；进一步地，令o1针对每题采样64条推理链并取多数投票，其正确率提高到约83%，若对1000条候选答案用评分函数重排序则可达93%[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=are no longer effective at,for the USA Mathematical Olympiad)。

  可见，多样性采样结合投票显著提升了复杂推理题的**可靠性和准确度**[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=are no longer effective at,for the USA Mathematical Olympiad)。这种方法以**增加推理次数**为代价换取性能增益，是TTS策略的基本形式之一。

- **自我评估与验证（反思式校正）**：系统2推理强调在中间步骤进行**自我检查**，及时发现并纠正错误，从而避免“一条路走到黑”。实现这一点的方式多种多样。例如，

  - Alibaba的**Marco-o1**模型在推理过程中周期性地插入“**等等！也许我犯了错误，我需要从头重新考虑**”这样的反思提示。这会促使模型暂停当前解题路径，对先前推理进行审视，识别潜在谬误并调整思路[venturebeat.com](https://venturebeat.com/ai/alibaba-researchers-unveil-marco-o1-an-llm-with-advanced-reasoning-capabilities#:~:text=Another key innovation in Marco,and refine its thought process)。这样的**自我质疑机制**让模型充当自己的审稿人，显著提高推理链的可靠性。
  - 在**Tree-of-Thoughts (ToT)** 思维树框架中，模型会对每一步产生的候选“想法”打标签评估其前景：标记为“确定（sure）”、“可能（maybe）”或“不可能（impossible）”达到目标[promptingguide.ai](https://www.promptingguide.ai/techniques/tot#:~:text=To perform BFS in ToT,The process is illustrated below)。通过这一**自评打分**，模型能在搜索过程中**提前淘汰不可能成功的分支**，聚焦于有希望的路线。这相当于在解题时不断检查部分解是否合理：若某步被判定走入死胡同，模型便回溯并尝试其他路径。
  - OpenAI的o1模型由于接受了大量基于反馈的强化训练，也学会了**识别并修正自身错误**的本领[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=Similar to how a human,This process)——遇到困难时它会调整策略或纠正前序步骤中的疏漏，从而使推理链逐步完善。

  综上，自我评估与验证机制通过**引入反馈回路**，赋予模型类似人类“检查工作”的能力，极大减少了推理过程中错误的累积。

- **小模型辅助/裁判（协同推理）**：为了降低大型模型推理时的成本，有一种策略是**引入辅助的小模型**来参与部分推理或结果评估。在一些方案中，小模型扮演**工具或裁判**的角色。例如，

  - 可让一个轻量模型先尝试解子问题或提供初步思路，大模型再基于这些线索展开深度推理；
  - 或者在大模型生成答案后，调用小模型对答案进行**审核评估**，给出正确性判断作为反馈。

  这样的协作有助于减少大模型不必要的计算，也为结果增加一道保险。一个相关思路是利用**Mixture-of-Experts（MoE）架构，将大型模型拆分为多个小专家模块，各模块专长于不同类型的问题，推理时按需激活对应专家**而非每次动用整套大模型[ibm.com](https://www.ibm.com/think/news/deepseek-r1-ai#:~:text=ablations and hyperparameter searches%2C” says,Soule)。DeepSeek的R1模型据报道就采纳了类似理念，在训练和推理中有效减少计算量，却表现出与OpenAI o1相当的推理能力[ibm.com](https://www.ibm.com/think/news/deepseek-r1-ai#:~:text=Why all the buzz%3F A,use%2C according to the company)[ibm.com](https://www.ibm.com/think/news/deepseek-r1-ai#:~:text=ablations and hyperparameter searches%2C” says,Soule)。具体来说，DeepSeek团队宣称R1在数学、编程等基准上达到与o1相媲美的成绩，但由于架构和优化上的高效设计，**使用成本降低约96%**[ibm.com](https://www.ibm.com/think/news/deepseek-r1-ai#:~:text=Why all the buzz%3F A,use%2C according to the company)。这表明，通过**小模型协作或模块化专家机制**，**可以大幅提升推理效率**。总的来说，小模型辅助/裁判是一种实践中平衡算力与性能的工程技巧：充分利用小模型（小模块）快速、廉价的优势，处理简单事务或监督大模型，从而实现“1+1>2”的效果。

- **触发机制（动态调控推理深度）**：在实际应用中，系统2/TTS策略往往需要设计**触发条件**，决定何时采用额外的推理步骤或策略。例如，

  - 并非所有问题都需要多轮深度思考，如果模型对某问有十足把握，直接输出即可；
  - 但当**模型不确定**或检测到冲突时，就应触发更谨慎的系统2流程（如自洽重采样或树搜索）。
  - 不确定性的信号可以来自模型对自身答案的置信度评分、不同采样结果之间的一致性、或验证模块的反馈。

  举例来说，

  - 在上述Marco-o1的机制中，模型遇到可能出错的迹象（比如推理链矛盾）时，会通过预设的提示语主动进入“重新思考”模式[venturebeat.com](https://venturebeat.com/ai/alibaba-researchers-unveil-marco-o1-an-llm-with-advanced-reasoning-capabilities#:~:text=Another key innovation in Marco,and refine its thought process)，这是一种**显式触发**策略(这里，Marco-o1 并没有做“显式的程序化矛盾检测”，而是模型会**周期性地自我提示**进入“重新思考”模式，以重新评估、识别潜在错误并细化思路)。
    - 具体，Marco-o1 的“反思触发”放在它的**MCTS + 推理动作策略**里一起使用：
      - 它把思维过程离散成 step / mini-step（32/64 个 token）作为搜索的“动作”单元；在这些**动作边界上**可以插入反思提示，让模型回溯或重写该段思路（属于“显式触发”）[arXiv](https://arxiv.org/html/2411.14405v1)。
      - 与此同时，它并行用到一种**置信度驱动的树搜索**：用生成 token 的**对数概率与前 5 个候选的相对关系**去计算**每条 rollout 的均值置信分**，据此扩展/筛枝（但这不是“矛盾检测器”，而是一种**信心评分**和分支选择的启发式）[arXiv](https://arxiv.org/html/2411.14405v1)。
    - 所以，“如何检测到推理链矛盾？”——在 Marco-o1 的公开材料里，并**没有**给出像“数值约束冲突、单位不一致、公式残差不为 0”这类**规则化检测**；而是：
      - **靠提示显式触发**：在固定/周期的**step 或 mini-step 边界**插入“我可能错了，需要重想”的**反思提示**，由模型**自评**并重写之前的链路（这是“触发”的核心做法）；
      - **靠搜索的“信心”信号**：对当前分支的**平均置信分**较低时，MCTS 会倾向于**换枝/回溯**，间接起到“这条路可能有问题”的效果（但仍属于**概率启发**而非显式矛盾判定）
  - 另外在ToT树搜索中，每当某一步被模型评估为“impossible”时，就触发了**回溯**和**分支切换**[promptingguide.ai](https://www.promptingguide.ai/techniques/tot#:~:text=To perform BFS in ToT,The process is illustrated below)，避免浪费算力在无解路径上。

  未来还可能引入更加自动化的触发机制，例如根据**概率分布熵值**来判断模型不自信，从而动态决定是否调用自我验证或多样性思考流程。良好的触发机制可以让系统2策略按需介入、收放自如，在保证准确率的同时尽量减少不必要的计算开销。

<br>

## 3. RL 如何让模型“会推理”

### 1) 从“提示驱动”到“策略驱动”：RL 训练推理策略

不同于只模仿人类解答的监督学习，RL 让模型通过**优化最终结果的奖励**来**自己发现**有效的推理策略。在 RL 微调中，我们把模型视作**智能体**：它生成一串推理步骤（chain-of-thought, CoT），并依据**最终答案是否正确**获得奖励。当**只**奖励“答对”时，模型会被**激励出任何能通向正确答案的内部思考过程**，哪怕这种过程与人类不同。实践表明，这种面向结果的训练能显著解锁高级推理行为：

- **传统 CoT（提示工程）**：通过提示语诱导模型“写出步骤”，但**推理机制没变**，仍是一次性左到右生成，缺少全局搜索/自检能力。性能提升主要来自提示诱导与样例迁移。
- **OpenAI o1 与早期 RL 推理模型：** OpenAI 的 *o1* 系列率先大规模使用 RL 提升 CoT 推理。o1 **通过 RL 训练**强化了隐式的逐步推理能力，而不仅仅是“下一个词”的预测；这证明 RL 能在“人类示例”之外进一步塑造 LLM 的解题能力。
- **思维链的“自发涌现”：** 一个重要发现是，即便**没有显式教**模型“每步该怎么想”，**只**用“终局答对给分”的 RL，模型也会**自发学会写出细致的思维链**。因为在优化奖励的过程中，模型“意识到”**更系统、更冗长的思考**能提高正确率，于是开始**自检、回溯、换路**等——这些都**不在奖励函数里**，却自然出现了。换言之，**CoT 成为一种涌现的解题策略**：模型为了拿到更高奖励，会**更长、更有结构地思考**。

> 直观理解：**RL 把“会不会认真想、什么时候该回头重来、怎么换思路”这些行为学成了“策略”**；TTS 则是在测试时给这套策略**更多试错与搜索的预算**。

### 2) 奖励来自哪里：Outcome vs Process

训练“会推理”的策略，需要**可学习的奖励信号**：

1. **Outcome 终局奖励**（答对给分）：简单可扩展，但**信用分配稀疏**、容易“会抄不会想”。
2. **Process 过程奖励**（逐步打分 PRM）：对**每一步**推理判对/判错，“细粒度”奖励更稳定；OpenAI 的 **Let’s Verify Step by Step** 证实**过程监督**（PRM）比单纯终局监督更可靠，提供了 **PRM800K** 数据集与显著的数学题解决率。[arXiv](https://arxiv.org/abs/2305.20050)
3. **RLAIF（AI 反馈）**：用强模型/判别器替代人类打分——**RL from AI Feedback**，在多任务上**与 RLHF 相当**，缓解人工标注瓶颈，也常用来**训练 PRM/Verifier**。[arXiv](https://arxiv.org/abs/2309.00267)

> 实务要点：若任务可程序化验证（数理/代码），可把**可执行验证**（单元测试、符号计算）转成**过程或终局奖励**；否则用 **PRM/RLAIF** 近似“过程正确”。

### 3) 如何用 RL 训练思维链（R1 示例化流程）

- **任务与奖励：** 明确目标（多步数学/理工/多跳问答），定义奖励=**最终答案是否正确**。不对中间步骤打分，避免人为偏置。
- **输出格式：** 约定 `<think>…</think>` + `<answer>…</answer>`，确保训练时**确实生成思维链**。
- **RL 训练环：** 用 GRPO 等算法对同题抽多次尝试、相对排序、更新策略。**无需人工标注**；仅靠自动判定最终答案正误推进学习。
- **监控涌现行为：** 观察思维链长度、结构、自反思/回溯等是否出现，并据此调节长度上限、奖励约束等。
- **可选风格精修：** 若纯 RL 模型准确但不够友好，可加少量 SFT 或偏好奖励，使其**既会想又好用**。

#### 详细个案：DeepSeek-R1 Zero/R1

*DeepSeek-R1*（2025）是一个**从 RL 出发训练推理能力**的开源 LLM 代表。它展示了仅靠**面向结果的 RL**也能解锁接近/超越人类示范的“系统2”式 CoT。

- **只用“终局奖励”的 R1-Zero：** 团队以 DeepSeek-V3 Base 为起点，**不做**人类解答的监督微调，直接进入 RL（使用 **GRPO**）阶段：

  - **奖励=答对 1，答错 0**。
  - 输出格式用 `<think>…</think>` 写思考、`<answer>…</answer>` 给答案，但**不对中间步骤打分**。这相当于只给两条规则：“按规定格式作答、答对就加分”，把**“怎么想”**的空间最大化交给模型。

- **技能快速跃迁：** 纯 RL 的 *R1-Zero* 在 AIME-2024 高难数学集上，**准确率从 ~15.6% 飞升到 ~77.9%**；再配合测试时的自洽投票：

  - **带随机性的解码采样**：对同一提示/题目，进行 $k$ 次独立生成（常用温度/Top-p 随机采样；R1 论文只明确“采样多次”，未公开具体温度/Top-p 细节）。[arXiv](https://arxiv.org/pdf/2501.12948)
  - **抽取与规范化答案**：从每条思维链中抽取最终结论（如数值、代数式），做单位/格式规范化，得到可比较的候选答案。
  - **多数表决**：统计每个候选出现频次，**取众数作为最终答案**（论文称 *majority voting / self-consistency decoding*；未说明平票时的专用规则，一般做法是再以似然或首个命中的候选作为回退策略，但 R1 论文未具体给出）。

  这使得模型性能跃升最高到 **~86.7%**。编程竞赛、理工科问答等领域也显著提升。这表明 **RL 真正把“会思考”学进了模型**。

- **“自进化”式行为涌现：** 训练推进时，R1-Zero 会**主动拉长思维链**（等于给自己更多“思考时间”）；它学会插入“**wait**”等**自我反思**语句，暂停复核；学会**自检中间步骤、卡住就换路**。这些高级策略完全是**为拿奖励而自发产生**，并非预设规则。正如 *Nature* 论文中强调：**“无需教它如何解题；只要激励恰当，它会自主发展出比人类教学更先进的策略。”**

- **超越只看人类示范：** R1-Zero 在数学、编程、STEM 等**可验证**任务上，**超过**仅靠人类示范监督训练的对手；并可**蒸馏**到更小模型延续推理力。这说明**纯 RL 能突破“模仿上限”**：不受人类样例的偏差限制，模型能探索出**更优**的推理路径。

- **多阶段精炼成最终 R1：** R1-Zero 的答题风格有时冗长或语言混杂。因此团队采用**多阶段流水线**：

  - ① 先用少量高质对话对V3 base模型做**冷启动 SFT** （作为RL的actor网络），改善可读性与指令遵循；这解决 R1-Zero **语言混杂、可读性差**带来的**早期 RL 不稳定**。冷启动**加快收敛**，并把“可读”的人类先验写进策略先验。

    - 先收集**几千条**“可读、长 CoT”的高质量数据，对 DeepSeek-V3-Base 做一次**小规模 SFT**，目的是把“会想且好读”的回应风格**注入到初始策略**，作为后续 RL 的起点。
    - 数据来源包括：少样例长 CoT 提示、直接让模型生成“反思+校验”的详细答案、筛选 R1-Zero 输出并**人工后处理**，统一成**可读模板**：`|special_token|<reasoning_process>|special_token|<summary>`（前者是 CoT，后者是面向用户的总结）。

  - ② 再**Reasoning-oriented RL（面向推理的第一轮 RL）**，在保证推理力的同时维持语言质量；这让模型在“**仅终局正确**”的硬约束下**自发涌现系统2行为**（长链思考、反思、策略切换），同时用轻量的语言一致性项**约束可读性**。

    - **奖励**以“**答对=1/答错=0**”为主（规则/可执行奖励）；为缓解多语混杂，**额外加一个“语言一致性奖励”**（CoT 中目标语占比），**与正确性直接相加**形成最终奖励。

  - ③ 随后**Rejection Sampling + 大规模 SFT混入非推理数据**，扩知识面与通用能力；这一步也**为小模型蒸馏**准备了体系化的高质量语料。

    - 当②阶段的推理 RL 收敛后，用该 checkpoint 做拒绝采样收集新的 SFT 数据，并混入非推理类数据（写作、事实问答、自我认知、翻译等），对 V3-Base 再做一轮大规模 SFT（约 80 万条：其中推理约 60 万、非推理约 20 万，训练 2 个 epoch）。数据的采样主要分为两个部分：
      - **推理数据**：针对推理提示多次采样→**只保留判定为“正确”的链路**；为增强可读性，**过滤掉混合语种答案、超长段落、代码块等不友好 CoT**；对难以规则判定的样本，引入**“生成式奖励模型”**（把 ground truth与预测喂给 DeepSeek-V3 做判别）辅助筛选。
      - **非推理数据**：**复用** DeepSeek-V3 的 SFT 数据管线；某些通用任务在回答前使用V3模型**提示生成潜在 CoT**，但对非常简单的问候类**不强制 CoT**，避免无谓冗长

  - 最后**RL for All Scenarios**（第二轮 RL，兼顾对齐与推理），把“强推理”与“好交互/安全性”**在同一 RL 框架下联合优化**：既维持系统2能力，又使输出更**有用、无害、可读**。

    - 在③的大规模 SFT 后，做第二轮 RL，**一边继续强化推理**，一边**对齐人类偏好**（**helpfulness/harmlessness**），得到最终 **DeepSeek-R1**。

      - **推理数据**：延续 R1-Zero 的**规则化奖励**（Rule-based rewards, 数学/代码/逻辑正确性）。

      - **通用数据**：引入**偏好奖励模型**（PRM, 基于 DeepSeek-V3 的1)偏好配对与 2)prompt 分布）进行**偏好对齐**。具体来说，这把大量“同一提示下的两段回答（A 胜、B 负）”作为训练样本，用**Bradley–Terry/逻辑回归式的成对损失**训练一个打分器 $r_\phi(x,y)$，使它对“被偏好”的 $y_w$ 打更高分，对“不被偏好”的 $y_l$ 打更低分。

      评估细节上：

      - **helpfulness** 只看**最终给用户的summary** 评估“是否回答了用户需求、是否相关、是否清晰”。（避免干扰底层推理过程），
      - **harmlessness** 检查**整段输出**（含 CoT+summary）以发现潜在风险与偏见。

      在 RL 里，这两路打分与其它可编程奖励（如果该样本也属于可验证任务）会**线性加权**成总奖励，例如（只是示例，文中并没有提到是如何integrating rewards和diversing data distribution的）：
      $$
      r_{\text{total}} = \alpha\cdot r_{\text{task}} + \beta\cdot r_{\text{help}}(\text{summary}) + \gamma\cdot r_{\text{safe}}(\text{full})
      $$
      然后用 **GRPO** 做组相对更新（同一 prompt 采样数个候选，分高者相对分低者获得正优势）。DeepSeek-R1 的多阶段管线（拒采+大规模 SFT + 二次 RL）正是基于此把“强推理”与“人类偏好对齐”合到一起的。
    

### 4) RL 学“何时加算力”：把 TTS 变成策略（Compute-as-needed）

- **问题**：TTS 很强，但代价高。
- **思路**：把“**是否触发长思考/扩展多少搜索宽度**”做成**动作**，设计**延迟-准确**的混合奖励（如：答对 +，超预算 −）。RL 由此学到**动态算力分配策略**，把“**何时长想**”学进模型（训练时可用 PRM/Verifier 作为**即时反馈**）。
- **证据**：关于“最优地扩展测试时算力”的研究指出，**测试时搜索/重采样的收益-成本**可被系统建模与优化；结合**密集的过程打分（PRM/verifier）**能更稳定地指导“何时继续想”。[google scaling law](https://arxiv.org/abs/2408.03314)

### 5) 把 TTS 的“思考经验”学回去：蒸馏/偏好优化 × RL

- 用 TTS 阶段得到的**高质量推理链/投票结果**，做 **蒸馏/偏好优化**，把“强检索+搜索”的行为**压回小模型**，形成“**便宜的系统2近似**”。例如，R1作者将在③ 阶段准备好的800K数据在多个开源中小模型上进行蒸馏（这里蒸馏的细节没有太给出，也没有给出防止模型灾难性遗忘的方法）。这一做法只涉及SFT和偏好优化， 而不涉及RL。
- 过程路线：**(1) TTS 产链 → (2) PRM/Verifier 评估 → (3) 选优/打分 → (4) RL/PO 蒸馏**。许多近作（R-PRM 等）也在奖励模型中把“**推理-验证-优化**”闭环在一起。[arXiv](https://arxiv.org/html/2503.21295v1?utm_source=chatgpt.com)

### 6) 最近RL Reasoning 工作

#### SWiRL：逐步 RL（Step-Wise RL）用于多步 QA 与数学 [9]

只用“终局奖励”的 RL 在**长链任务**上反馈极稀疏，学习慢。Google DeepMind + Stanford 提出 **SWiRL**：把长推理**分解为若干步骤**并进行**逐步的 RL 更新**。做法是迭代产出模型求解轨迹（可包含工具调用），把整条解答切成**子动作**，并为**每个子动作**打**部分奖励**（是否朝正确方向推进）。这样模型无需苦等“最后对错”，而是**每步都有信号**，可更快学会“如何把对的步骤串起来”。

切分子动作的三种“粒度控制”手段

1. **格式驱动（首推）**：强制 `<step>` 标签；每步必须选择一个 `<tool>`（包括 NONE）。
2. **语义触发**：正则或浅解析检测**等式/关键词**（“因此/所以/接着/计算/设/令/所以得到/查找”）自动开新步。
3. **定长切片**：每 N token 切一刀，最简单也最通用，适合早期模型或无清晰语义边界的任务；与 1) 结合更稳。

举例说明：

- 一般任务模板

  ```
  <step i>
  <thought>...当前推理...</thought>
  <tool> NONE | SEARCH | CALC | CODE_RUN | RETRIEVE ... </tool>
  <args>{...工具参数或为空...}</args>
  <obs>...（环境返回的观测，由系统填充）...</obs>
  </step i>
  ...
  <final>...最终答案...</final>
  ```

- 数学

  - 每出现一个独立的等式/运算或“所以/因此/接着”，开启新 `<step>`；
  - 或者更简单：**每 N 个 token**强制切一步（例如 N=32），保证最少的分步粒度（对早期模型更稳）。

  **部分奖励怎么设计？（不需要过程标注）**设题目有**已知正确最终答案** $y^*$。在第 $i$ 步，解析当前 `<thought>` 中**最新的等式**或**数字结论**（若没有就 $r_i=0$）：

- 代码

  每个 `<step>` 产出**一个最小变更**（如一个函数/补丁片段），`<tool>=CODE_RUN` 触发测试，`<obs>` 返回**通过测试数**与错误日志。

**效果：** 在 HotpotQA（两跳 QA）上，SWiRL 较一跳 RL 基线**提升 12–15%**；在 GSM8K、MuSiQue、ComplexWebQ 等上也有显著增益（部分达 **21.5%+ 相对提升**）。更有意思的是，**跨任务泛化**：仅在 HotpotQA 上训练，**zero-shot**到 GSM8K 也有 **+16.9%** 的提升。这说明逐步 RL 训练出的**通用解题技能**（规划多步、串联子解）可迁移到新任务。

> 注：SWiRL 的“中间奖励”接近“过程监督”的精神，但这里的中间信号主要由**模型自生成轨迹**自动派生，无需人工标注；整体仍属 RL 范畴。



#### Satori：通过 RL 学“何时继续/反思/换路”的自调度推理 [10]

MIT/SUTD/Harvard的另一方向是让模型**自己决定何时多想、何时反思、何时换策略**。*Satori*（2025）为 7B 模型显式加入“推理动作”，并通过 RL 教会其使用。

- 模型可输出 `<|continue|>`（继续当前思路）、`<|reflect|>`（暂停自检）、`<|explore|>`（换一条路）等**元认知动作**，作者将这一模式定义为chain of Action Thought (COAT), 并声明CoT是COAT的一个特例。

- **格式微调（Format-Tuning SFT）**

  - **目的：** 教模型掌握系统2推理的“格式与动作接口”。

  - **做法：**
    - 用少量高质量样例（带 `<|reflect|>`, `<|continue|>`, `<|explore|>` 等标签）进行 SFT。
    - 只训练**动作结构与输出样式**，不追求高推理正确率。
    - 结果：模型能流畅地产生带“动作标记”的思维链，这是 PPO 的初始策略。

-  **强化学习主循环（PPO）**

  - **核心目标：** 在格式固定的基础上，用强化学习优化“何时反思/换路/终止”。

  - **奖励基础：**

    - 主奖励为“最终答案是否正确”（可程序化 0/1）。

    - 为防稀疏奖励，引入两项关键机制：

      - **RAE（Restart & Explore）**：状态增强；

        - **动机：** 初期正确轨迹稀少，模型大部分从题面出发都会失败，导致训练样本过于集中、奖励稀疏。

          **做法：**

          - 从模型生成的推理轨迹中提取**中间状态（intermediate partial trajectories）**；
          - 在这些状态后加上 `<|reflect|>` 动作，形成新的起点集合；
          - 这些起点连同原数据集 $D$ 一起组成**增强数据分布 $D_{\text{restart}}$**；
          - 在 PPO 训练中交替从 $D$ 与 $D_{\text{restart}}$ 采样（**与原始的prompt作为一次训练的起点）**，进行新的 rollouts。

          **作用：**

          - 等价于一种“**在线数据增强**”，扩展原始任务分布；在原文中，作者称之为自我提升（Self-improvement）。
          - 让模型能在多样化的中间状态下学习反思与纠错；
          - 显著提升 RL 的样本利用率与状态覆盖度。

      - **PB（Preference Bonus / ORM）**：密集奖励。

- 训练中引入“**重启-探索**”技巧：允许在中途从早期状态**回滚重试** (但是不会直接回退到最开始，知识)，并周期性**重启 RL** 以跳出局部最优。

**结果：** Satori 学会了**自主管理推理过程**：遇到不一致就 `<|reflect|>` 修正早先步骤，走不通就 `<|explore|>` 改路。一个 7B 模型在数学基准上达到了**同级最强**甚至**追平/超越 30B** 的表现，核心就在于它学会了**不过度在错误路线上深挖**。

> 要点：RL 不仅能学“写思维链”，还可学一套**灵活的推理策略**（何时多想、何时自检、何时换路），显著提升复杂问题的可靠性。



#### RL 算法细化（以 SGPO 为例）[11] [12]

R1 用的 **GRPO** 有个痛点：训练早期若一批采样**全错**，优势值全为 0，策略**不更新**。**SGPO（Stepwise Guided Policy Optimization）** 用“**判别器/裁判**给予**部分信用**”的思路缓解：即使最终错误，但**较好的中间步骤**也能获益，从而**“从错误中学习”**。理论上它收敛更快、实践中早期增益更明显。类似改进还包括负样本/正样本的信用分配加权、降低回合成本的“无训练”式 RL、小语句式自评信号纳入奖励等——总体目标是让**长视野推理的 RL 微调更稳、更省、更有效**。



#### ARPO[12]

**问题背景**：**多轮工具使用**（搜索/代码执行/浏览器）会在**收到外部反馈后**显著提高模型下一个片段的生成不确定性（token 熵升高）。现有“整条轨迹级”的RL（如 GRPO/DAPO）更多比较整条 rollout，**忽视了工具调用后的“高不确定性时刻”**；这既浪费探索预算，也学不牢“每一步如何用工具”的细粒度行为。[arXiv](https://arxiv.org/pdf/2507.19849)

**核心想法**：在传统整轨采样之外，**在“高熵”的工具调用步“分叉”做局部采样**（partial rollout），把探索火力集中到不确定性最大的关键步；再配合一种**优势归因估计**（Advantage Attribution Estimation），把不同分叉段的优势信号“分账”，让策略能真正**内化“哪一步的工具用得好/不好”**。

算法要点

- 熵驱动的自适应 Rollout（Entropy-based Adaptive Rollout）

  - 先做 **N 条全局轨迹**（global sampling），统计每条在“工具调用后前 k 个 token”的初始熵 $H_{\text{initial}}$；随后在每次工具反馈后，再生成 k 个 token 得到该步熵 $H_t$，计算**归一化熵变**：
    $$
    \Delta H_t=\mathrm{Normalize}(H_t-H_{\text{initial}})
    $$
    若 $\Delta H_t$ 大，说明该步不确定性上升，需要加大探索。

  - 以
    $$
    P_{i,t}=\alpha+\beta\cdot\Delta H_{i,t}
    $$
    作为**分叉概率**，若 $P_{i,t}>\tau$，就在该步对第 $i$ 条轨迹**分出 Z 条局部分叉**（partial rollouts），补入采样集合。这样在总预算 $M$ 内动态平衡“整轨 vs 局部”探索。伪代码见 **Algorithm 1, 行 16–18**。

> 直觉：把“工具后高不确定时刻”当成 **beam 的扩展位置**，用少量预算把搜索面铺开；其余时刻保持常规整轨采样，**精确地“花在刀刃上”**。该机制把每次 rollout 的复杂度从 $O(n^2)$ 降到 $O(n\log n)$～$O(n^2)$ 区间（取决于超参）。

- 优势归因估计（Advantage Attribution Estimation）

  - **Hard 版本**：对“共享前缀”的 token 用**同一优势**，对“分叉后的各自 token”用**各自优势**；优势以分组标准化回报计算。

  - **Soft 版本（默认）**：直接用 **GRPO 目标**做“组内平均优势 + token-level 比率”优化：
    $$
    J_{\text{GRPO}}=\mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\big(r_{i,t}A_{i,t},\ \mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_{i,t}\big)-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})\Big]
    $$
    当两条分叉**共享同一前缀**时，它们的 $r_{i,t}$ 相等，从而实现对共享段的**一致加权**；分叉后的 token 则得到**各自的**比率与优势，等价于**在目标里隐式区分“共享 vs 分叉”**。

> 论文给出一个“宏动作（macro-action）版”**广义策略梯度（GPG）定理**，说明在 Transformer 上用“分段（partial rollout）”进行优化是有理论根据的：把输出切成若干段 $MA_T$，仍可写成

$$
\nabla_\theta J=\mathbb{E}_\tau\sum_T\nabla_\theta\log\pi_\theta(MA_T\mid MS_T)\,A_T(\tau),
$$

​	把单 token PG 当作特例。

- 奖励与训练范式

  - **奖励**：延续 Tool-Star 的层级奖励：正确性 + 格式 + 多工具协作加分 $r_M$（若既用 <search> 又用 <python>）。错误格式记 −1。公式见式 (8)。

  - **训练范式**：冷启动 SFT（学习工具格式/行为），再进行 RL（ARPO/GRPO 比较一致的设置），评测覆盖数学、知识推理与深搜索 13 个基准。

从 OpenAI o1、DeepSeek-R1 到 SWiRL、Satori，**“强化推理”**正快速走向成熟：用更小或更少监督的模型，在数学、编程、多跳科学问答等最难基准上拿到 **SOTA 级**表现。



<br>

## 4. 总结：TTS策略的优势、代价与应用展望

综上所述，将系统2思维风格与测试时算力相结合的推理策略，为大型语言模型带来了显著的性能提升和能力拓展。这种**“花时间换准确率”**的做法具有以下主要优势：

- **更强的推理能力与准确率：** 通过在推理过程中进行搜索、验证和多样化尝试，模型在复杂任务上的表现**大幅超越**以往单次直出答案的水准[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=are no longer effective at,for the USA Mathematical Olympiad)[arxiv.org](https://arxiv.org/pdf/2305.10601#:~:text=non,Code repo with)。许多以前棘手的问题（如高难度数学、推理问答、规划搜索等）现在都能被模型攻克或显著改进。模型的推理过程更接近人类专家的解题思路，甚至在某些专业基准上达到或超过人类水平[arxiv.org](https://arxiv.org/abs/2502.17419#:~:text=Foundational Large Language Models ,like cognitive abilities)。对于追求高可靠性的应用（如科学分析、代码生成、医学问诊），系统2策略带来的**准确率提升**非常宝贵。
- **推理过程的透明性与可验性：** 系统2风格强调显式的中间推理，这使AI的决策过程更加透明，可供审查和验证[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=without retraining base models 3,simple and complex tasks efficiently)。通过记录模型的思维链或搜索树，人类或算法可以**检查每一步推理的合理性**，发现谬误并进行干预[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=without retraining base models 3,simple and complex tasks efficiently)。这种**可解释性**在安全关键领域尤为重要：我们可以定位模型哪里出了问题并加以纠正。此外，推理记录还为评估模型行为提供了宝贵数据，有助于进一步改进模型或构建**链式评估**机制，让模型自我监督。
- **计算与性能的灵活权衡：** 引入TTS后，我们获得了一个调节模型性能的全新“旋钮”。通过增加或减少推理时的搜索宽度/深度，可以在**响应速度**与**答案质量**之间灵活折中[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=* Trade,more time and mental effort)。例如，面对简单任务或实时性要求高的场景，我们可以采用较少的采样和推理步骤以保证效率；而在遇到关键的复杂问题时，则可以投入更多算力让模型充分思考，获得尽可能准确的结果[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=* Trade,more time and mental effort)。这种**按需加速**的能力，类似于人类遇到难题时“花更多时间仔细考虑”[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=,more time and mental effort)。它突破了仅靠提示工程调优模型的局限，使得**无须重新训练**就能通过增加推理计算来提升性能[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=This scalability is a significant,compute resources at test time)。在大模型规模增长遇到边际效益递减的背景下，这种推理层面的扩展为持续提高AI能力提供了一条新路径[venturebeat.com](https://venturebeat.com/ai/alibaba-researchers-unveil-marco-o1-an-llm-with-advanced-reasoning-capabilities#:~:text=scaling laws,time scaling)。

当然，系统2 + TTS策略也伴随着一定的**代价和限制**，需要辩证看待其在实际落地中的价值：

- **计算成本与时延：** 最明显的代价是推理开销的增加。让模型多想几遍、多走不同路线，势必消耗更多的算力和时间。上文提到的o1例子中，为了提升准确率而进行了数十甚至上千次采样，这对实时应用来说几乎不可行。即使较温和的自洽CoT（如5-10次采样）也将使推理成本提高数倍。因而在大规模应用场景下，如何在性能和成本间取得平衡是重要考量。如果对响应时延要求严格（如在线对话或交互系统），系统2方法可能显得太慢——有研究指出，模型逐步反思会导致回答延迟**几秒到几分钟**不等[ibm.com](https://www.ibm.com/think/news/deepseek-r1-ai#:~:text=which produced an answer without,a “chain of thought” manner)。因此目前这类策略更适合**离线分析**、**低并发高价值**的任务，或通过优化（如并行化采样、异步搜索）来减轻时延。
- **实现复杂度与可靠性：** 将上述机制融入推理流程，需要精心设计算法和基础架构支持。多轮交互、调用子模型、工具执行等都会增加系统复杂性，带来潜在的故障点。举例来说，引入小模型评判就要求维护两个模型的协同；树搜索需要管理分支和回溯逻辑。这些都对工程实现提出了更高要求。此外，虽然系统2策略整体提升了模型可靠性，但仍**不能保证绝对正确**。模型可能在多数思路上反复犯相似错误（导致投票无效），或自我评估时出现偏差。触发机制若拿捏不当，也可能出现要么不触发导致错误漏检，要么过度触发徒增算力浪费的情况。这提醒我们，当前的系统2方法仍在探索优化，**尚未成熟到可无缝处理一切问题**。对于某些有明确算法可循的问题（如精确数学计算），或许直接使用符号计算更有效；而对于那些无明确标准的开放场景，模型即便穷尽思考也未必拿得出令人满意的答案[venturebeat.com](https://venturebeat.com/ai/alibaba-researchers-unveil-marco-o1-an-llm-with-advanced-reasoning-capabilities#:~:text=However%2C many applications involve open,world challenges%2C” Alibaba researchers write)。
- **应用价值与局限：** 系统2+TTS策略在需要高可信度和复杂推理的任务中展现出巨大价值，例如科研助理、代码验证、决策支持等。这些领域错误代价高，宁可牺牲一些速度也要保证正确性。然而，在日常对话、内容生成等对**速度和流畅度**要求更高的应用中，大多数用户期望即时响应，可能难以接受模型“深思熟虑”的等待时间。此外，某些场景对**答案的一致性**或**创意**要求高，但对逻辑严谨性要求不高（比如文学创作），系统2的优势就不明显，反而可能因为过度求稳导致输出缺乏多样性。最后，从商业角度看，投入额外算力也意味着更高费用，不是所有服务都能承担。因此，TTS技术更适合作为**可选的增强模式**：当任务确实需要、且用户愿意等待时再启用系统2推理，否则仍采用快速的系统1模式。正如近期业界所认识到的，我们才刚开始探索**推理时扩展**的无限可能[venturebeat.com](https://venturebeat.com/ai/alibaba-researchers-unveil-marco-o1-an-llm-with-advanced-reasoning-capabilities#:~:text=scaling laws,time scaling)。随着研究的推进，我们有望找到更高效的算法和更智能的调控，让模型能根据任务难度**自适应地切换**快思考或慢思考。这将进一步降低系统2策略的使用门槛，扩大其实际应用范围。

**总而言之**，系统2推理结合测试时算力扩展，为强化大模型的思维能力提供了一条富有前景的道路。它让AI不再局限于训练时期学到的静态映射，而是赋予其一定程度的**“自主思考”**能力——在给定新问题时可以探索、反思、举一反三。这种范式正引领着新一波推理模型的竞赛：从OpenAI o1、DeepSeek R1到各大公司的创新模型，无不围绕如何在推理阶段“多花功夫”以取得突破。而工程实践中，我们需要权衡其带来的效益和成本，选择最恰当的场景加以应用。可以预见，在未来的AI系统中，“快系统1”与“慢系统2”有望融合为**灵活混合的架构**[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=This combination of System 1,2 processing offers several advantages)：平时以直觉快速响应，遇到复杂问题时调用深度推理模块，达到既高效又可靠的智能。这种汲取了人类认知优点的混合策略，将有助于打造更加强大且可信的AI助手，为实际应用创造更高的价值。

<br>

## Reference

1. Zhong-Zhi Li *et al*. *“From System 1 to System 2: A Survey of Reasoning Large Language Models.”* arXiv preprint (2025)[arxiv.org](https://arxiv.org/abs/2502.17419#:~:text=> Abstract%3AAchieving human,like cognitive abilities)[arxiv.org](https://arxiv.org/abs/2502.17419#:~:text=Foundational Large Language Models ,like cognitive abilities)
2. OpenAI. *“Learning to reason with LLMs.”* OpenAI Research blog (Sep 2024)[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=Similar to how a human,on several difficult problems below)[openai.com](https://openai.com/index/learning-to-reason-with-llms/#:~:text=are no longer effective at,for the USA Mathematical Olympiad)
3. Aili McConnon. *“DeepSeek’s reasoning AI shows power of small models, efficiently trained.”* IBM News (Oct 2025)[ibm.com](https://www.ibm.com/think/news/deepseek-r1-ai#:~:text=Reasoning models became the hot,a “chain of thought” manner)[ibm.com](https://www.ibm.com/think/news/deepseek-r1-ai#:~:text=Why all the buzz%3F A,use%2C according to the company)
4. Ben Dickson. *“Alibaba researchers unveil Marco-o1, an LLM with advanced reasoning capabilities.”* VentureBeat (Nov 2024)[venturebeat.com](https://venturebeat.com/ai/alibaba-researchers-unveil-marco-o1-an-llm-with-advanced-reasoning-capabilities#:~:text=OpenAI o1 uses “inference,in tasks with standard answers)[venturebeat.com](https://venturebeat.com/ai/alibaba-researchers-unveil-marco-o1-an-llm-with-advanced-reasoning-capabilities#:~:text=Another key innovation in Marco,and refine its thought process)
5. Johannes Koeppern. *“Self-Consistency with Chain of Thought.”* Medium (Aug 2023)[medium.com](https://medium.com/@johannes.koeppern/self-consistency-with-chain-of-thought-cot-sc-2f7a1ea9f941#:~:text=ADVANTAGEDESCRIPTIONQUANTIFICATIONImproved PerformanceSelf,more comprehensive understanding of complex)
6. Shunyu Yao *et al*. *“Tree of Thoughts: Deliberate Problem Solving with Large Language Models.”* NeurIPS 2023[arxiv.org](https://arxiv.org/pdf/2305.10601#:~:text=non,Code repo with)[arxiv.org](https://arxiv.org/pdf/2305.10601#:~:text=The literature on human cognition,example%2C research on reinforcement learning)
7. DAIR AI. *“Tree of Thoughts (ToT) - Prompt Engineering Guide.”* (2023)[promptingguide.ai](https://www.promptingguide.ai/techniques/tot#:~:text=To perform BFS in ToT,The process is illustrated below)
8. Gloqo AI. *“System 1 vs System 2 in Modern AI: From AlphaGo to LLMs.”* (2023)[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=* Trade,more time and mental effort)[gloqo.ai](https://www.gloqo.ai/insights/combining_system_1_and_system_2_thinking/#:~:text=without retraining base models 3,simple and complex tasks efficiently)
9. Goldie, A., Mirhoseini, A., Zhou, H., Cai, I., & Manning, C. D. (2025). *Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use.* In *Conference on Language Modeling (COLM 2025).* https://doi.org/10.48550/arXiv.2504.04736
10. Shen, M., Zeng, G., Qi, Z., Hong, Z. W., Chen, Z., Lu, W., ... & Gan, C. (2025). Satori: Reinforcement learning with chain-of-action-thought enhances llm reasoning via autoregressive search. *arXiv preprint arXiv:2502.02508*.
11. Lee, H., Jo, D., Yun, S., & Kim, S. (2025). SGPO: Self-Generated Preference Optimization based on Self-Improver. arXiv preprint arXiv:2507.20181.
12. Dong, G., Mao, H., Ma, K., Bao, L., Chen, Y., Wang, Z., ... & Dou, Z. (2025). Agentic reinforced policy optimization. *arXiv preprint arXiv:2507.19849*.



















