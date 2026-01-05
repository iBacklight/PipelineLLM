# Day 5 持续学习 Continual Learning

[TOC]

我们在之前的学习中已经了解了LLM后训练的基本流程。

## 1.了解持续学习

初识大模型，我们都惊叹于它丰富的知识储备和清楚的表达方式。然而，世界与业务持续变化（例如新事实、新术语、新任务，新工具等），如果每次都全量重训模型会使得代价过高；同时社会规范、企业政策与用户偏好也在持续演化；还要兼顾安全与合规。如此看来，我们需要让达模型有自主持续更新的能力，既要更新知识库和业务能力，又要保证符合人类偏好和安全。这就是持续学习的作用。

我们定义持续学习（Continual Learning, CL）：用于在一个**任务/分布按时间到来**的序列中，用**同一个模型**持续学习新东西，同时**尽量不丢**旧的能力（避免灾难性遗忘，catastrophic forgetting），并且控制计算与存储成本。




## 2. 典型CL方法

一般来说，我们有三大CL方法家族：

- **Replay-based**：保留一小部分旧数据/表征做“复习”（经验回放、exemplar、生成式回放等）。
- **Regularization/Constraints**：对参数或输出加**稳定项**限制漂移（如 **EWC** 的参数重要性正则、**GEM** 的梯度投影、蒸馏/KL 到参考模型）。
- **参数隔离/结构扩展**：为不同任务引入专门的模块（Adapters/LoRA/MoE/多头），减少相互干扰。

### 2.1 Replay-based（回放/复习）

**做什么**：在学新任务时，把一小部分旧数据（或旧任务的“代表性表征”）混入训练，像复习提纲一样稳住旧能力。

**常见形态**

- **Memory buffer（经验回放）**：维护一个小缓存，每轮从中均匀或按策略采样复习样本。
  - 选样策略：随机 / **herding**（靠近类均值）/ 基于**损失或梯度范数**（难例优先）/ **多样性**（k-means、k-center）。
  - 流式维护：**reservoir sampling**（数据流不定长时保持等概率代表性）。
- **Exemplar（原型/代表集）**：为每类或每任务挑 K 个“最能代表”的样本，推理时也可配**最近均值分类器（NMC）**来增强稳态。
- **Latent/Feature replay（表征回放）**：存**中间特征**而非原始样本，节省存储并弱化隐私风险。
- **Generative replay（生成式回放）**：用生成模型（或旧版 LLM 本身）“造”旧任务样本做复习；可避开直接存原数据，但质量不稳时会**漂移累积**。

<br>

#### 2.1.2 Experience Replay (ER) / Rehearsal

ER[7]的核心思想是,在任务/数据流到来时，用一个**很小的记忆库**保存少量旧样本；训练新数据时，把**新样本与记忆库样本混合**组成小批次一起训练，从而“复习”旧知识，缓解遗忘。具体来说，我们可以设定

- 数据按任务或数据流顺序到来：$\mathcal{D}_1,\mathcal{D}_2,\dots$（单遍 / 单次见样本）。
- 记忆库容量固定：$K$（可按“每类 $m$”或“总量 $K$”分配）。
- 模型：标准的分类网络 $f_\theta$。

对每个到来的小批新数据 $\mathcal{B}_{\text{new}}$：

1. **从记忆库采样**旧样本：$\mathcal{B}_{\text{mem}}\sim \text{Sample}(\mathcal{M})$（常用与新批同等大小或按比例 $\alpha$）。

2. **混合批训练**：$\mathcal{B}=\mathcal{B}_{\text{new}}\cup\mathcal{B}_{\text{mem}}$。

   - 标准交叉熵（多类单头）：
     $$
     \mathcal{L}(\theta)=\frac{1}{|\mathcal{B}|}\sum_{(x,y)\in\mathcal{B}}\mathrm{CE}(f_\theta(x),y).
     $$

   - 反向传播、更新 $\theta$。

3. **更新记忆库**：把当前新批的一部分样本写入 $\mathcal{M}$。

> 常用技巧：混合比控制“新:旧”= 1:1 或 2:1；类增量时对采样做**类均衡**，减轻新类偏置。

**记忆库如何维护**?

1. **Reservoir Sampling（蓄水池采样）**：面对**未知长度数据流**的均匀抽样。第 $t$ 个样本到来时，以概率 $\min(1, K/t)$ 放入；若内存已满，被选入时随机替换已有样本。这样能保证**所有历史样本被等概率保留**。

   - 伪代码：

     ```python
     if |M| < K: M.append(x)
     else:
         j = randint(1, t)
         if j ≤ K: M[j] = x
     ```

2. **Ring Buffer / FIFO**：按队列先进先出（简单，但对早期样本不公平）。

3. **每类均衡（class-balanced）**：为每个已见类分到近似相等的名额（类增量更稳）。

4. **代表性选样**（可选）：Herding / K-Center / K-Means，让记忆覆盖更有代表性的点。

**优缺点**

- ✅ 极其简单、基线：少量样本就能显著抑制遗忘。
- ❌ 需要保留旧样本（隐私/合规场景可能受限）;记忆库小 → 对**分布漂移/长尾**敏感；

#### 2.1.2 Incremental Classifier and Representation Learning (ℹCaRL)

Incremental Classifier and Representation Learning (iCaRL) [3] 是一种用于增量持续学习的方法，简单来说就是教会模型“复习 ”。我们在**类增量（class-incremental）**场景中，模型会按阶段接收新类别的数据，此时需要：

- 记住旧类别（抗遗忘）
- 学好新类别（可塑性）
- 在**同一模型、同一头**上对所有已见类别进行推断（没有任务 ID）

iCaRL在解决这些问题上有三个核心设计：

1. **表征学习 + 近均值分类器（NCM）**

- 用一个特征提取器 $\phi_\theta(x)$ 学表征；

- 推断时**不用线性分类头**，而用“类别均值最近”（Nearest Class Mean, NCM）：
  $$
  \mu_y=\frac{1}{|P_y|}\sum_{p\in P_y}\phi_\theta(p),\quad
  \hat{y}=\arg\min_y \|\phi_\theta(x)-\mu_y\|_2 .
  $$

- $P_y$ 是类别 $y$ 的样本代表集（exemplar set）。

2. **蒸馏 + 二元交叉熵（BCE）联合训练**

- 训练新阶段时，把**旧模型在新数据上的输出**作为旧类别的软目标（蒸馏）以保留旧知识；

- 对新类别用真实标签学习。整体用**多标签式 BCE**（对每个类别独立 sigmoid），减轻“新类过度挤压旧类”的偏置。让我们记 $z_c=f_\theta^c(x)$ 为类别 $c$ 的 logit，$p_c=\sigma(z_c)$；新类真标签 $y_c\in\{0,1\}$，旧模型软目标 $q_c^{\text{old}}\in[0,1]$。

  **新类监督项（BCE）：**
  $$
  \mathcal{L}_{\text{new}}(x)=-\sum_{c\in\mathcal{C}_{\text{new}}}
  \big[\,y_c\log p_c+(1-y_c)\log(1-p_c)\big].
  $$
  **旧类蒸馏项（BCE ）：**
  $$
  \mathcal{L}_{\text{distill}}(x)=-\sum_{c\in\mathcal{C}_{\text{old}}}
  \big[\,q_c^{\text{old}}\log p_c+(1-q_c^{\text{old}})\log(1-p_c)\big].
  $$
  **总损失（可加权）：**
  $$
  \mathcal{L}(x)=\mathcal{L}_{\text{new}}(x)\;+\;\lambda\,\mathcal{L}_{\text{distill}}(x).
  $$

- 直观理解：**旧类用蒸馏保持输出分布，新类用真标签推动学习**。

3. **记忆库（Examplar buffer）与“放牧（Herding）”选样**

- 全局记忆容量 $K$ 固定；当已见类别数为 $t$ 时，每类分到 $m=\lfloor K/t \rfloor$ 个样本；样本为原始数据（例如图片，prompt原文等），而不是经过旧模型提取的特征。

- 为逼近**特征空间中的类均值**，逐个挑选能最接近整体均值的样本（贪心），形成 $P_y$；

  - 首先计算均值$\mu = \frac{1}{n}\sum_{x}\phi(x)$;

  - 然后通过不断迭代获得$P_y$的第k个样本$p_k$ (注意，在第 $k$ 步我们要选第 $k$ 个样本 $p_k$，目标是让**前 $k$ 个已选样本的均值**尽量接近真实类均值 $\bar \mu$)
    $$
    p_k = \arg\min_{x}\big|\big|~\bar \mu - \frac{1}{k}\big(\phi(x)~+~\sum^{k-1}_{j=1}\phi(p_j)\big)\big|\big|
    $$

  - 其中， $\phi(x)$是从全局记忆中被要选中的第k个样本，$P_y$ = ($p_1, p_2,...,p_{m}$)

- 每个阶段结束后，用**当前模型的特征**重算均值，必要时**截断**每类样本到 $m$。

- 这就是**Herding（放牧选样）**：在**每个阶段**里，针对每个已见类别 $y$，用当前特征 $\phi_\theta$ 从该类训练样本中**按顺序**挑出“最能逼近类中心”的样本序列。

- **记忆库（Exemplars / memory buffer）**：真正**存进内存**用来回放的集合，等于把每类的 $P_y$**截断到**前 $m=\lfloor K/t\rfloor$ 个后再取并集：
  $$
  \text{Memory}=\bigcup_{y}\, P_y[0:m]
  $$

我们举个例子来彻底走一边iCaRL的流程，我们的设定

- **全局记忆容量**：K = 8
- **阶段1**：已有两类 A、B ⇒ 每类配额 $m=\lfloor K/2\rfloor=4$
- **阶段2**：新增类 C ⇒ 共3类 ⇒ 每类配额 $m=\lfloor K/3\rfloor=2$

为便于演示，给出各类的**二维特征**（只是便于心算的玩具坐标）：

- **类A**：全部候选特征值：$(0,0),(1,0),(0,1),(1,1),(2,2)$，类均值 $\bar \mu_A=(0.8,0.8)$
- **类B**（设定一组集中在 (3,3) 附近的点）：全部候选特征值：$(3,3),(3,2),(2,3),(4,4),(2,2)$，类均值 $\bar \mu_B=(2.8,2.8)$
- **类C**（第二阶段才出现，集中在 (−1,−1) 左右）：全部候选特征值：$(-1,-1),(-1,0),(0,-1),(-2,-2),(0,0)$，类均值 $\bar \mu_C=(-0.8,-0.8)$

**阶段1（类 A、B）**

1）训练（无蒸馏）

- 数据：只用 A、B 的训练集（尚无旧模型）。

- 头：逐类 sigmoid（多标签 BCE）。

- **损失（只有新类监督项）**：
  $$
  \mathcal{L}(x)=
  \sum_{c\in\{A,B\}}
  \big[-\,y_c\log p_c-(1-y_c)\log(1-p_c)\big]
  $$

  > 此时没有旧类 ⇒ 没有蒸馏项。

训练完得到当前模型 $f_\theta$ 和表征 $\phi_\theta$。

2）Herding 选样（每类取前 m = 4 个）

- 类 **A** 的 herding前 4：**逐步选(保证前k个更加贴近均值)**

  - **步 1（k=1）**
     $\text{target}_1 = 1\cdot(0.8,0.8) = (0.8,0.8)$
     看谁离 (0.8,0.8) 最近？显然 **(1,1)** 最近 ⇒ $p_1=(1,1)$
  - **步 2（k=2）**
     $\text{target}_2 = 2\cdot(0.8,0.8) - (1,1) = (0.6,0.6)$
     候选：$(1,0),(0,1),(0,0),(2,2)$。
     离 (0.6,0.6) 最近的是 **(1,0)** 或 **(0,1)**（对称）。任选其一，比如 $p_2=(1,0)$。
  - **步 3（k=3）**
     $\text{target}_3 = 3\cdot(0.8,0.8) - [(1,1)+(1,0)] = (0.4,1.4)$
     候选：$(0,1),(0,0),(2,2)$。
     离 (0.4,1.4) 最近的是 **(0,1)** ⇒ $p_3=(0,1)$。
  - **步 4（k=4）**
     $\text{target}_4 = 4\cdot(0.8,0.8) - [(1,1)+(1,0)+(0,1)] = (1.2,1.2)$
     候选：$(0,0),(2,2)$。
     离 (1.2,1.2) 最近的是 **(2,2)** ⇒ $p_4=(2,2)$。

  **得到 A 类的 Herding 顺序（前 4 张）**：
  $$
  P_A = \big[(1,1),\ (1,0),\ (0,1),\ (2,2)\big]
  $$

- 类 **B** 的 herding（同样的方法，过程略）前 4：
   $P_B=[(3,3),(3,2),(2,3),(4,4),\ldots]$

3）形成记忆库（每类 m=4）
$$
\text{Memory}^{(1)}=P_A[0\!:\!4]\ \cup\ P_B[0\!:\!4]
$$
4）重新计算类均值用于推断（NCM）：用**当前表征值** $\phi_\theta$ 对记忆库样本求类均值

- $\mu_A=\frac{(1,1)+(1,0)+(0,1)+(2,2)}{4}=(1.0,1.0)$
- $\mu_B=\frac{(3,3)+(3,2)+(2,3)+(4,4)}{4}=(3.0,3.0)$

5）阶段1推断/测试（NCM示例）

假设测试点 $x=(0.9,0.7)$：

- 到 $\mu_A$ 的距离：$\sqrt{(0.9-1)^2+(0.7-1)^2}=\sqrt{0.01+0.09}=0.316$
- 到 $\mu_B$ 的距离：$\sqrt{(0.9-3)^2+(0.7-3)^2}$ 明显更大
   ⇒ 判为 **A**。

训练结束，把当前模型冻结成 **旧模型 $f_{\text{old}}$**，留作下一阶段蒸馏用。

**阶段2（新增类 C）**

1）准备训练集

- **新类数据**：C 的训练样本（很多）
- **旧类回放**：阶段1的记忆库样本（A与B各4张）
- 训练集 = 新类数据（主角） + 旧类 exemplars（旧知识锚点）

2）生成蒸馏软目标（对旧类 A、B）：用**冻结的旧模型** $f_{\text{old}}$ 对训练集**所有样本**前向，得到其对旧类的概率
 $\{q_A^{\text{old}}, q_B^{\text{old}}\}$。（对 C 类没有旧目标，因为旧模型不认识 C。）

3）优化（BCE + 蒸馏 BCE）：让我们记 $z_c=f_\theta^c(x)$ 为当前要训练的模型对类别 $c$前向的 logit，$p_c=\sigma(z_c)$；新类真标签 $y_c\in\{0,1\}$，旧模型软目标 $q_c^{\text{old}}\in[0,1]$。

- **新类监督项（只对 C）**：
  $$
  \mathcal{L}_{\text{new}}(x)_{c\in{C}}=
  -\,y_c\log p_c-(1-y_c)\log(1-p_c)
  $$

- **旧类蒸馏项（对 A、B）**：
  $$
  \mathcal{L}_{\text{distill}}(x)=
  \sum_{c\in\{A,B\}}
  \big[-\,q_c^{\text{old}}\log p_c-(1-q_c^{\text{old}})\log(1-p_c)\big]
  $$

- **总损失**：$\ \mathcal{L}=\mathcal{L}_{\text{new}}+\lambda\,\mathcal{L}_{\text{distill}}$

> **直观例子（单样本 x 的一条损失）**
> 假设 x 是 A 类的 exemplar：旧模型给 $q_A^{\text{old}}=0.9, q_B^{\text{old}}=0.1$。而新模型当前输出 $p_A=0.8, p_B=0.15, p_C=0.05$。那么该样本的损失（只列出**蒸馏项**）：
>  $\mathcal{L}_{\text{distill}}= -[0.9\log0.8+0.1\log(1-0.8)] -[0.1\log0.15+0.9\log(1-0.15)]$。
> 它推动新模型在 **A/B 两维**上贴近旧模型的行为，从而**守住旧知识**。（而对 C 类的真标签监督会主要来自 C 的新数据样本。）

训练完得到更新后的当前模型 $f_\theta$ 与表征 $\phi_\theta$。

4）记忆库重分配与**截断**

当前类数 $t=3$ ⇒ 每类配额 $m=\lfloor 8/3\rfloor=2$。

- **A**：保留 $P_A$ 的前 2 个 ⇒ $[(1,1),(1,0)]$，无需重新计算
- **B**：保留 $P_B$ 的前 2 个 ⇒ $[(3,3),(3,2)]$，无需重新计算
- **C**：用最新 $\phi_\theta$ 做 herding，取前 2 个（按我们给的数据选接近 $\bar g_C$ 的点，如 $[(-1,-1),(-1,0)]$，示意）

新的记忆库：
$$
\text{Memory}^{(2)}=\{A\!:\!2\}\cup\{B\!:\!2\}\cup\{C\!:\!2\}\quad(\text{共6张}\le K)
$$

> 说明：不一定必须用满 K；常见做法是严格平均到 $\lfloor K/t\rfloor$ 每类。

5）重算类均值（用于阶段2之后的推断）

- $\mu_A=\frac{(1,1)+(1,0)}{2}=(1.0,0.5)$
- $\mu_B=\frac{(3,3)+(3,2)}{2}=(3.0,2.5)$
- $\mu_C = \frac{(-1,-1)+(-1,0)}{2}=(-1.0,-0.5)$

6）阶段2推断（NCM示例）

假设测试点1：$x=(0.9,0.7)$

- 距 $\mu_A=(1.0,0.5)$：$\sqrt{0.1^2+(-0.2)^2}=0.224$
- 距 $\mu_B=(3.0,2.5)$：明显更远
- 距 $\mu_C=(-1.0,-0.5)$：更远
   ⇒ 判为 **A**。

测试点2：$x=(-0.9,-0.6)$

- 到 $\mu_C=(-1.0,-0.5)$：$\sqrt{(-0.1)^2+0.1^2}=0.141$
- 到 $\mu_A,\mu_B$：更远
   ⇒ 判为 **C**。

**优缺点**

- ✅ 通常**有效**缓解遗忘；学习顺序不影响性能。
- ❌ 存储（需要额外的储存和运算）/隐私合规压力；样本选择与维护需要工程投入。
- 💡 LLM 小贴士：可回放**指令+理据（rationale）**、**负样本**、**工具调用轨迹**，比仅回放输入输出更稳。



<br>

### 2.2 Regularization / Constraints（正则与约束）

不用或少用旧数据，直接在参数或输出层面给“稳定项”，限制学习新任务时的“漂移”。

#### 2.2.1 Elastic Weight Consolidation (EWC)

**直觉**：旧任务“关键参数”不应该被取代。

EWC[3]首先识别对训练完成模型的最优参数（或者说最重要的参数/权重），然后选择性地减慢这些参数/权重的学习速度，从而达到在学习新任务的时候保护已经学习的知识的效果。具体做法是，在旧任务最优参数 $\theta^*$ 周围加二次惩罚，并使用**Fisher 信息矩阵（Fisher Information Matrix）**来衡量各个网络参数对以学习任务的重要性：
$$
\mathcal{L}(\theta)
= \mathcal{L}_{\text{new}}(\theta)
+ \frac{\lambda}{2}\sum_i F_i\,(\theta_i-\theta_i^*)^2
$$

- 对角线$F_i$ 估计, 表示当前训练参数 $\theta_i$ 的重要性。

  - 在旧任务数据$D_A$的每一个样本$x$上，求参数$\theta*$的对数似然 $\nabla_{\theta_i}\log p(y|x;\theta^*)$， 再求它的平方期望。

  - 在实际计算的时候，其实是取近似的方法，因为FIM本身计算很耗费时间。EWC 为了降低计算量，只保留 Fisher 的对角项：
    $$
    F_i \approx 
    \mathbb{E}_{x \sim D_A}
    \left[
    \left(
    \frac{\partial \log p(x|\theta_A^*)}{\partial \theta_i}
    \right)^2
    \right]
    $$

  - 注意：这里**不是把参数“放到对角线上”**，而是我们把所有参数**展平成一个向量** $\theta = [\theta_1,\dots,\theta_n]$。（见下面代码的`_flatten_params`函数）

- **LLM 场景**：旧任务可指令集合；$\mathcal{L}_{\text{new}}$ 为 SFT 交叉熵或自监督 loss。

**优缺点**：

- ✅ 不需保存旧数据（或只需很少用于估计 $F$）；实现轻量。
- ❌ 对角近似**过度简化**(损失精度以换取可以接受的计算开销)；需要新任务和旧任务有相似处；跨很多任务时**惩罚累积**导致很难学会新任务。

**代码实现**

这里需要解释的是，EWC 的 Fisher 近似是（将上面带期望的公式展开成）：
$$
F_i \approx \frac{1}{|D_A|}\sum_{x\in D_A}
\left(\frac{\partial}{\partial\theta_i}\log p(x|\theta_A^*)\right)^2
$$
这里的 $\log p(x|\theta)$ 是**对数似然**。对于分类任务，若模型输出 logits $z=f_\theta(x)$，则
$$
p(y|x;\theta) = \mathrm{softmax}(z)_y,
\qquad
\log p(y|x;\theta) = \log\mathrm{softmax}(z)_y
$$
所以一般可以用log_softmax`函数来写`。同时，我们知道交叉熵损失定义为
$$
\text{CE}(x,y) = -\log p(y|x;\theta)
$$
所以
$$
\nabla_\theta \log p(y|x;\theta) = -\,\nabla_\theta\,\text{CE}(x,y)
$$
平方后负号消失。因此 **我们可以直接用 CE 的梯度平方** 来近似 Fisher。

```python
import torch, copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F

@torch.no_grad()
def _flatten_params(model): # 把所有参数展平成一个向量 
    return parameters_to_vector([p.detach() for p in model.parameters()])

def estimate_fisher_diagonal(model, dataloader_A, num_batches=100, device="cuda"):
    """
    返回：
      fisher_diag_flat: 按参数展平后的向量（与parameters_to_vector一致的顺序）
      theta_star_flat:  旧任务最优参数（展平向量）
    """
    model.eval()  # 关掉dropout/BN的训练态，但允许梯度计算
    theta_star_flat = _flatten_params(model)
    # 准备累加器（与参数同形的扁平向量）
    fisher_diag_flat = torch.zeros_like(theta_star_flat, device=device)

    n_seen = 0
    it = iter(dataloader_A)
    for _ in range(num_batches):
        try:
            x, y = next(it)
        except StopIteration:# sampels not enough
            it = iter(dataloader_A)
            x, y = next(it)

        x, y = x.to(device), y.to(device)

        # 需要梯度，不能用no_grad
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        logits = model(x) 
        loss = F.cross_entropy(logits, y, reduction="mean")
        # 如果使用普通log_softmax, 则
        # log_probs = F.log_softmax(logits, dim=-1)
		# log_likelihood = log_probs[torch.arange(y.size(0)), y].mean() #对二维矩阵 log_probs 做逐样本选列的操作
        # | 样本       | 取出的元素             |
		# | ------- ---| -----------------    |
		# | 第 0 个样本 | 第 0 行第 `y[0]=0` 列 |
		# | 第 1 个样本 | 第 1 行第 `y[1]=2` 列 |

		# 注意：这里 log_likelihood = E[log p(y|x;θ)]，所以loss要加负号
		# loss = -log_likelihood 
        loss.backward()

        # 将每个参数的grad展平并平方累加
        grads = []
        for p in model.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p).flatten())
            else:
                grads.append(p.grad.detach().flatten())
        g_flat = torch.cat(grads) # 把模型里每一层的梯度展平成一个大向量并拼接起来
        fisher_diag_flat += g_flat.pow(2)

        n_seen += 1 # 这里使用小样本作近似平均期望，而非在整个数据集上求平均

    fisher_diag_flat /= max(n_seen, 1)
    return fisher_diag_flat, theta_star_flat

def ewc_penalty(model, theta_star_flat, fisher_diag_flat, lambda_ewc=1.0):
    theta_flat = parameters_to_vector(model.parameters())
    # 元素级二次项：sum_i F_i (theta_i - theta_i^*)^2
    quad = (fisher_diag_flat * (theta_flat - theta_star_flat).pow(2)).sum()
    return 0.5 * lambda_ewc * quad

# ==== 训练循环（新任务 B）==== 举例如何加入这个penalty
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
lambda_ewc = 2.0 # 超参

for x, y in dataloader_B:
    x, y = x.to(device), y.to(device)
    model.train()
    optimizer.zero_grad()

    logits = model(x)
    loss_new = F.cross_entropy(logits, y, reduction="mean")

    loss_reg = ewc_penalty(model, theta_star_flat, fisher_diag_flat, lambda_ewc)
    loss = loss_new + loss_reg

    loss.backward()
    optimizer.step()

```

> 相关：**SI、MAS** 等也是“参数重要性”正则的变体，区别在重要性度量的定义。

#### 2.2.2 Learning without Forgetting（LwF）/知识蒸馏 / KL 到参考模型

**直觉**：输出行为别飘。

**做法**：Learning without Forgetting（LwF）[4]：微调加知识蒸馏（教师模型输出的软标签给学生模型学习，提升学生模型的性能和泛化能力，并且缓解灾难性遗忘）。具体如下做法如下，假设：

- 老模型（教师）为 $f_{\text{tea}}$（参数冻结），新模型（学生）为 $f_{\text{stu}}$；
- 新任务带标签的数据为 $\mathcal{D}_{\text{new}}=\{(x,y)\}$（**无需旧数据**）

对任意输入 $x$，教师与学生的**温度化**输出分别是
$$
p^{(T)}_{\text{tea}}(x)=\text{softmax}\!\left(\frac{z_{\text{tea}}(x)}{T}\right),\quad
p^{(T)}_{\text{stu}}(x)=\text{softmax}\!\left(\frac{z_{\text{stu}}(x)}{T}\right),
$$
其中 $z(\cdot)$ 为 logits，$T>1$ 让分布“变软”，显露类间相对关系（比如，类别中有猫，狮子，大象；输入是猫，当温度大于1，输出会显示：依然是猫80%，但是狮子18%， 大象2%，也就是说，输入比起大象更像狮子）。

我们首先来看什么是蒸馏损失。蒸馏损失常用等价写法有两种（注意常见做法会乘上 $T^2$ 做梯度尺度补偿）：
$$
\mathcal{L}_{\text{distill}}
= \alpha\, T^2 \cdot
\text{CE}\!\big(p^{(T)}_{\text{tea}},\, p^{(T)}_{\text{stu}}\big)
\;\;\;\; \text{或}\;\;\;\;
\alpha\, T^2 \cdot
\text{KL}\!\big(p^{(T)}_{\text{tea}} \,\|\, p^{(T)}_{\text{stu}}\big)
$$
其中 $\alpha$ 控制蒸馏项的权重。这里要注意，在“用 CE 做蒸馏”时，我们优化的是**软标签交叉熵**：
$$
\text{CE}\big(p_{\text{tea}}^{(T)},\,p_{\text{stu}}^{(T)}\big)
= -\sum_i p_{\text{tea},i}^{(T)} \,\log p_{\text{stu},i}^{(T)}.
$$

它只在**学生分布**上取 $\log$。老师分布 $p_{\text{tea}}^{(T)}$ 只是**目标权重**（软标签），不参与求导，因此不需要 `log`。如果用 KL 形式：$\mathrm{KL}(p_{\text{tea}}^{(T)}\|p_{\text{stu}}^{(T)})=\text{CE}-H(p_{\text{tea}}^{(T)})$，后面那项是常数（对学生无梯度），所以优化上与 CE 等价。接下来，设批大小为 B，序列长度为 T，有效 mask：$m[b,t]\in\{0,1\}$（有效=1，padding/被忽略=0），有效 token 总数$N_{\text{eff}}=\sum_{b,t} m[b,t]$。LwF的损失包括三个部分：

**蒸馏项（软标签）**——类别维要保留求和：
$$
\mathcal{L}_{\text{distill}}
= -\frac{1}{N_{\text{eff}}}\sum_{b,t} m[b,t]\;
\sum_{i=1}^{V} \underbrace{p_{\text{tea}}^{(T)}(b,t,i)}_{\text{老师软分布}}
\;\log \underbrace{p_{\text{stu}}^{(T)}(b,t,i)}_{\text{学生软分布}}
\;\;\;(\times\,T^2)
$$
**监督项（硬标签）**——one-hot 折叠后无类别求和：
$$
\mathcal{L}_{\text{sup}}
= -\frac{1}{N_{\text{eff}}}\sum_{b,t} m[b,t]\;
\log p_{\text{stu}}(b,t,y[b,t])
$$


相当于整个batch的损失就是把所有 **mask=1** 的位置的 $-\log p_{y}$ 求和，再除以那些位置的数量 $N_{\text{eff}}$。需要注意的是，由于目标是离散标签 $y$ 的 one-hot 分布，在 **硬标签** 情况下，类别内求和被 one-hot 折叠成“取真类概率”，因此**没有 $p_{\text{tea}}$**（老师分布）

接下来的正则项
$$
\mathcal{L}_{\text{reg}} = \lambda \cdot \Omega(\theta)
\quad(\text{如 } \ell_2 \text{、dropout、或参数重要性正则})
$$
**总损失**

$$
\mathcal{L}= \alpha\mathcal{L}_{\text{distill}}+ (1 -\alpha)\mathcal{L}_{\text{sup}}+ \mathcal{L}_{\text{reg}}
$$

 $\alpha$ 控制蒸馏项的权重。

**优缺点**：

- ✅ 不留原数据也可做（只需老师推理）；对输出分布直接约束，**稳定可控**。
- ❌ 老师若有偏差会被“继承”；$\beta$ 取值需平衡稳态与可塑性。

**代码实现**

```python
import torch
import torch.nn.functional as F

def distill_loss(logits_stu, logits_tea, T=2.0, alpha=0.5, mask=None):
    """
    logits_*: [B, C] (分类) 或 [B, T, V] (LLM token 级)
    mask:     [B, T] 的0/1，有效token为1（分类可为None）
    """
    # 温度化分布
    p_tea_T = F.softmax(logits_tea / T, dim=-1)
	log_p_stu_T = F.log_softmax(logits_stu / T, dim=-1)
	loss = F.cross_entropy(logits_stu, p_tea_T, reduction='batchmean') * (T*T)
    
    # 温度尺度补偿
    dis_loss = alpha *(T * T) * kl
    return  dis_loss

def lwf_total_loss(logits_stu, y_new, logits_tea=None, T=2.0, alpha=0.5,
                   reg=None, mask=None, ignore_index=-100):
    # 新任务监督（正常温度）
    if logits_stu.dim() == 2:  # 分类
        sup = F.cross_entropy(logits_stu, y_new)
    else:  # LLM token 级: logits [B,T,V], y_new [B,T]
        sup = F.cross_entropy(
            logits_stu.view(-1, logits_stu.size(-1)),# 合并所有维度，最后是vocab_size
            y_new.view(-1),
            ignore_index=ignore_index # 忽略mask的位置
        )

    # 蒸馏项（可选，若提供了教师 logits）
    dist = 0.0
    if logits_tea is not None:
        dist = distill_loss(logits_stu, logits_tea, T=T, alpha=alpha, mask=mask)

    # 正则项（可选）
    reg_term = 0.0 if reg is None else reg

    return dist + (1 - alpha) * sup + reg_term

```



#### 2.2.3 GEM（Gradient Episodic Memory）及近似（A-GEM）

- **直觉**：更新梯度不应**恶化旧任务损失**。
- **做法**：令 $g$ 是新任务梯度，$\{g_k\}$ 是少量旧任务样本上的梯度，把 $g$ **投影**到满足 $\forall k,\; g^\top g_k \ge 0$ 的可行域（不增加旧损失）。等价于一个小型 QP 或其近似：

$$
\min_{g'} \frac{1}{2}\|g'-g\|^2 \quad
\text{s.t.}\; g'^{\top} g_k \ge 0\;\;\forall k
$$

- **A-GEM**：用**平均旧梯度**代替多个约束，单约束投影，计算更快。

**优缺点**：

- ✅ 明确“**不忘旧**”的几何条件；小量旧样本即可。
- ❌ 需要保留**旧样本梯度**；大模型上 QP/投影的**开销**与实现复杂度更高。

<br>

### 2.3 参数隔离 / 结构扩展（Adapters / LoRA / MoE / 多头）

**做什么：\**不给同一套参数强行“兼容一切”，而是\**为不同任务/域单独加模块**，互不干扰；推理时按需加载或路由。

**常见形态**

- **多头（multi-head）**：共享骨干，任务各自一个输出头；在分类/打分类任务清晰易用。
- **Adapters / LoRA（PEFT）**：冻结大模型主体，为新任务**挂小模块**（adapter 层、低秩 LoRA）；
  - 可做**每域一套 LoRA**，几乎零遗忘；
  - 也可用 **O-LoRA（正交约束）**、**LLaMA-PRO/块级扩展**等减少模块间干扰；
  - 需要**权重合并/选择器**支持部署（路由何时加载哪套 LoRA）。
- **MoE / 路由**：多专家结构，按输入路由到子网络；天然“软隔离”，但训练/部署更复杂（负载均衡、稀疏路由稳定性）。
- **Prompt/Prefix/IA³**：更轻量的参数高效方法；训练/切换成本低，适合“工具递增”。

#### 2.3.1 Packnet

Packnet [6] 同样适用于任务增量（task-incremental）持续学习。如果依次到来的任务 $T_1,T_2,\dots$，我们希望把多个任务都“装进”同一网络里，同时**不遗忘**已学任务。

**PackNet 的核心想法**：

- 在当前任务上训练整网 → **幅度剪枝**留下“关键权重” → **冻结这些权重**给该任务“占座”；
- 释放出来的“空位”（被剪掉的权重位置）**重新初始化**，专门给**下一个任务**训练；
- 循环往复，用**二值掩码**为每个任务记录其“子网络”，推理时按任务 ID 只激活对应权重子集。

我们来看看它的具体做法：

1. **剪枝与掩码**

- 对每层权重 $W$ 做**非结构化幅度剪枝**（常用 $|w|$ 作为重要性）：
  $$
  M = \mathbf{1}\{|W|\ge \tau\},\quad W \leftarrow W \odot M
  $$
  其中 $\tau$ 是分位阈值（例如保留前 $p\%$ 的大权重），$M\in\{0,1\}^{\text{shape}(W)}$ 是二值掩码。

- **冻结保留的权重**（对已学任务的掩码位置禁止更新，不过文章中提到这样做性能可能会有所下降，所以在冻结的权重上对原任务可能需要重新训练），**只训练未占用的位置**：
  $$
  \nabla W \leftarrow \nabla W \odot (1 - M_{1:t-1})
  $$
  即：第 $t$ 个任务训练时，仅更新前面任务未使用的位置。

2. **任务专属子网**

- 每个任务 $t$ 形成自己的掩码 $M_t$（可以与先前部分**不重叠**，或最小化重叠）。
- 推理时需**知道任务 ID**，使用 $\theta\odot M_t$ 进行前向（只激活任务对应的参数）。

3. **容量管理**

- 随任务数增加，**可用可训练参数**逐渐减少（被“占座”的越来越多）。
- 可设**分阶段剪枝率**（如每任务后保留 50~80% 大权重），在“性能”和“可用容量”之间权衡。
- 若容量耗尽，可：
  - 放宽剪枝率（牺牲旧任务一点性能换空间）；
  - 轻量扩容（增加层宽/附加块）；
  - 与 Adapter/LoRA 结合，将新任务更多转移到低秩或旁路参数。

**训练流程（每个任务）**

**输入**：任务序列 $\{T_1,\dots,T_K\}$，基模型 $\theta$。

**对任务 $t=1\ldots K$：**

1）**全网/子网训练**（只在当前**可训练位置**训练；只有对 $t=1$ 是全网训练）
$$
\min_\theta \ \mathbb{E}_{(x,y)\sim T_t}\big[\mathcal{L}_t(f_{\theta}(x),y)\big],\quad
\text{s.t.}\ \nabla\theta \odot \Big(1-\sum_{i=1}^{t-1}M_i\Big)
$$
2）**幅度剪枝**（对当前任务重要性排序），得到 $M_t$：

- 选择保留率 $r_t$（例如每层保留前 $r_t\%$ 的大权重）。
- $M_t=\mathbf{1}\{|W|\ge \tau_t\}$，$\tau_t$ 为按层阈值。

3）**冻结 & 打包**：

- 把被 $M_t$ 标记的位置视为“**属于任务 $t$** 的关键权重”，冻结它们；
- 被剪掉的位置（$1-M_t$）视作“**空闲位**”，在进入 $t{+}1$ 时重新初始化、供训练使用。

4）**评估/微调**：必要时在 $T_t$ 上微调冻结前的网络，巩固效果。

**推理**（给定任务 ID $t$）：输出 $f_{\theta\odot M_t}(x)$。

> 注：原版 PackNet 是单任务一头或多头均可，但需要**已知任务 ID**来选择掩码（task-incremental 协议）。

我们来看一个简单直观的例子：假设当前网络一层只有 10 个权重。

- 任务1训练后，按幅度保留前 60%（6 个）权重并冻结（成为 $M_1$）。
- 任务2只能在剩余 4 个“空位”上训练；完成后再在这 4 个里保留表现最好的若干（例如 3 个）作为 $M_2$，冻结。
- 任务3继续在剩余的 1 个“空位”上学（有些吃力了），可以选择放宽之前的剪枝率、扩容、或加 LoRA等来提升性能。

**优缺点**

- ✅ **强效**的防遗忘（**物理隔离**）；任务特定参数隔离；高存储效率。

- ❌ 复杂度较高；网络容量有限（可能会限制新任务的数量和性能）。

#### 2.3.2 MoE/Lifelong MoE

MoE（Mixture-of-Experts）[9]： 把 Transformer 的 FFN 层替换为一组并行的“专家”FFN，由一个门控（gate）对每个 token 选择少量专家做前向计算（通常 top-1 或 top-2）。这样**参数量可以很大**（很多专家），但**每个 token 只激活少数专家**，所以推理/训练的**FLOPs 近似与稠密 FFN 持平**。这是**稀疏模型(Sparse Model)**的直观实现（其余对应的是dense模型）。

![MoE](pics/MoE.png)

Figure 1 MoE 架构，图源: [NanoMoE](https://cameronrwolfe.substack.com/p/nano-moe)。图中展示了MoE在Transformer decoder层中的位置

MoE是怎么运作的呢？

1. **Router（门控/路由器）**

   - 对每一个 token（也就是经过注意力层后的一个向量表示 $h$），路由器会分别计算它与每个专家的匹配程度分数： $s_e=w_e^\top h+b_e$，然后我们得到每个token分数的概率 $p_e=\mathrm{softmax}(s_e)$。
   - 根据求出的概率选择 **top-k** 个专家（常见 k=1 或 2），如图1中示例：对 $x_1$ 选中了 FFN2；对 $x_2$ 选中了 FFN1、FFN2（并显示各自概率 $p=0.65, 0.8$ 的意思）。

2. **Experts（专家组）**

   - 本质上是一组**并行** **FFN**：$\{f_1,\dots,f_E\}$。

   - 每个 token 只会被送到被选中的少数几个专家里做前向（**稀疏激活**），其余专家不参与计算。

     - 这里要搞清一个概念，在多数 MoE 论文/实现里，“专家”并不等于“学科专家（例如心理学，生物学等）”。更贴切的理解是——**专家是对“当前 token 的表征特征”更擅长的一组 FFN**。这些特征往往是**词源/子词形态、语法角色、标点/括号/换行、数字形态、代码样式、语言片段**等“局部而高频”的模式，而不是整块的知识领域。

     - 我们通过一些例子来理解：下面这些都是在真实模型里常见的“专家偏好”，而且**与上下文强相关**（同一个词在不同上下文可能被送到不同专家）：

       - **形态/子词特征**：以 “-ed/-ing/-tion/-性/-化” 结尾的子词、前缀 “un-/re-/超/非”、阿拉伯数字“123”、十六进制“0x..” 等。
       - **符号/布局**：括号“()[]{}”、逗号/分号、Markdown/LaTeX 标记、换行/缩进、空白块。
       - **句法角色**：常见功能词（the, of, 的、了）、实体大写模式（PERSON/ORG）、句首/句尾位置特征。
       - **代码样式**：花括号+缩进、调用栈样式、关键词“def/class/return”、注释“# //”。
       - **语言片段**：同一个 subword 在英文 vs. 西语/中文环境里路由不同（受上下文词嵌入影响）。

       例如：

       - 输入 “if (count <= 0) { … }”，**括号、比较符**这类 token 的表征会激发“擅长代码符号模式”的专家；
       - 输入“比例为 12.5%”，包含 **数字+小数点+百分号** 的组合，常被送到偏好“数字/标点模式”的专家；
       - 输入“organization” ，但是在新闻语境与在生物论文语境，前后上下文不同，$h$ 不同，路由也可能不同——**不是按词面固定分派**，而是按**表征里的上下文特征**分派。

     - 专家的专业分配**不是先规定好的**。大多数 MoE 在训练开始时，路由器和各专家都**随机初始化**；哪位专家“像是在管数字/括号/时态/某类子词”等，都是**训练过程中自发形成**的，且会因随机种子、层次、数据分布与正则权重而不同。

3. **Dispatch & Combine（派发与聚合）**

   - 派发：把 token 复制/分片到被选中的专家输入队列（通常会有**容量限制**，防止某个专家过载）。

   - 聚合：专家的输出按路由权重 $\alpha_e$ 线性加权回收：
     $$
     y=\sum_{e\in\mathcal S_k(h)} \alpha_e(h)\, f_e(h),\quad
     \alpha_e(h)=\frac{p_e}{\sum_{j\in\mathcal S_k(h)}p_j}.
     $$

   - 之后接回残差，继续走后面的层。

4. **Load Balancing（负载均衡）**

   - 在 MoE 中，每个 token 只被分配给少数几个专家（Top-k）。如果没有约束，早期训练会出现“**赢家通吃**”：少数专家被频繁选中，其他专家几乎没梯度，导致容量浪费、模型不稳。
   - 为避免“热门专家拥堵”，训练时会加一个**均衡损失**（让各专家的使用率更均匀）。

**关键收益**

1. 参数规模 $\uparrow\uparrow$（更多专家）但每步激活专家固定（top-$k$），**算力近似不变**；
2. 专家可“分工”处理不同子分布，**天然适配多领域/多语言**；
3. 专家是**可扩展**的结构单元，便于后续增量。

#### 2.3.3 Lifelong MoE

MoE 解决了“容量”和“多分布建模”，但**直接微调**仍会遗忘。**Lifelong MoE** [8] 把 MoE 变成“**可阶段扩容、旧知识可冻结**、**更新可受控**”的 CL 工作流，核心在三件事：

1) 渐进式扩容（Progressive Expert Expansion）

- 当出现新分布/新领域/新时间片（阶段 $t$），**只新增少量专家**：$E^{(t-1)}\!\rightarrow E^{(t)}$，并保持 **top-$k$ 不变**。
- 由于每个 token 仍仅激活 $k$ 个专家，**训练/推理 FLOPs 基本不涨**，但模型“可表示的模式”变多了。
- 新专家初始化用“**Net2WiderNet 风格拷贝**”（从旧专家/路由复制并加小扰动），避免训练初期不稳定。

2) 冻结旧知识（Freeze Old Experts & Routing）

- 将阶段的**旧专家参数**和**旧路由分量**冻结，**不再被新的梯度覆盖**；
- 仅优化：**新增专家 $\theta_t$** 与**共享层（Attention/Dense）$\theta_d$**。
- 直观上，旧专家像“固化的记忆块”，新知识主要写入新专家，**物理隔离**减少灾难性遗忘。

3) 对旧模型蒸馏的稳定器（KL-to-old / Output Distillation）

- 用上一阶段模型 $M_{\text{old}}$ 的输出当“锚”，对当前模型 $M_{\text{new}}$ 施加 KL 正则，抑制输出分布漂移：
  $$
  \mathcal{L}= \underbrace{\mathbb{E}_{x}[\ell_{\text{LM}}(M_{\text{new}};x)]}_{\text{语言建模/任务损失}}+ \lambda\;\mathbb{E}_{x}\!\left[\mathrm{KL}\!\big(M_{\text{old}}(x)\,\Vert\,M_{\text{new}}(x)\big)\right].
  $$

- 优化变量只包含新增与共享部分：
  $$
  (\theta_t^*,\theta_d^*)
  =\arg\min_{\theta_t,\theta_d}\mathcal{L}\quad
  \text{s.t.}\quad \theta_{\text{old experts/gate}}\;\text{frozen}.
  $$

- 这等价于在“结构隔离”的基础上，再配一个“**输出层的软约束**”，双保险地稳住旧能力。

<br>

### 2.4 算法总览

| 方法族    | 代表                                       | 机制                | 需要任务ID           | 回放需求 | 遗忘控制                 |
| --------- | ------------------------------------------ | ------------------- | -------------------- | -------- | ------------------------ |
| 回放      | iCaRL/ER/GAN回放                           | 保留/生成旧样本复习 | 否（类增量时可单头） | 是       | 依赖样本锚点             |
| 正则/约束 | EWC、LwF、GEM                              | 软约束参数或梯度    | 否                   | 否       | 中等（易受任务差异影响） |
| 参数隔离  | **PackNet**、Piggyback、Adapters/LoRA、MoE | 硬/软划分参数子网   | **是**               | 否       | 强（几乎无干扰）         |

提到的算法对比

| 方法                       | 方法族                        | 是否需要旧数据                           | 需任务ID（推理）           | 典型评测头               | 核心机制                                       | 优点                              | 局限                                                |
| -------------------------- | ----------------------------- | ---------------------------------------- | -------------------------- | ------------------------ | ---------------------------------------------- | --------------------------------- | --------------------------------------------------- |
| **ER (Experience Replay)** | Replay-based                  | **需要**（少量样本/exemplars）           | 否                         | **单头**（类增量可用）   | 记忆库小样本 + 新旧混训                        | 简单强基线、实现成本低            | 隐私受限；记忆很小时对漂移敏感；新类偏置            |
| **iCaRL**                  | Replay + 原型(NCM) + 蒸馏     | **需要**（每类 m 个 exemplar）           | 否                         | **单头**（类增量）       | exemplars + 多标签BCE + 蒸馏；NCM近均值分类    | 类增量稳、对新类偏置更鲁棒        | 需存样本；herding/容量敏感；特征漂移需重算原型      |
| **EWC**                    | 正则/约束                     | **不需要**                               | 通常**要**（多头更常见）   | 多头更常见（单头可但弱） | Fisher 权重重要性约束参数漂移                  | 无需存数据，侵入小                | 任务跨度大时约束不足；λ/Fisher 估计敏感             |
| **LwF**                    | 蒸馏/约束                     | **不需要**（用旧模型对新数据蒸馏旧任务） | 否                         | 单头或多头               | 旧模型为旧任务提供软目标蒸馏                   | 无需旧数据、实现简单              | 类分布差异大时蒸馏不稳；仍会遗忘                    |
| **GEM**                    | 正则/约束（梯度投影）+ 小回放 | **需要**（少量记忆）                     | 训练时常用；推理可否       | 单头/多头均可            | 约束当前梯度不增大记忆样本损失（投影到可行域） | 强抗遗忘；理论清晰                | 计算重（需解投影）；内存/批次管理复杂               |
| **PackNet**                | 参数隔离/结构化               | **不需要**                               | **需要**                   | 多头或按掩码子网         | 任务后剪枝保留重要权重并冻结；空位训练新任务   | 遗忘极低；无需旧数据              | 依赖任务ID；容量渐耗尽；非结构化剪枝不友好硬件      |
| **LoRA（作CL适配）**       | 参数隔离/低秩适配             | **不需要**                               | **通常需要**（切换适配器） | 多头或共享头+适配器      | 主干冻结，每任务训练少量低秩适配参数           | 参量少、切换快、可与蒸馏/回放叠加 | 需要路由/任务ID；容量依旧有限，单头类增量需额外机制 |

<br>

<br>

## Reference

[1] Wu, T., Luo, L., Li, Y., Pan, S., Vu, T., & Haffari, G. (2024). Continual Learning for Large Language Models: A Survey. *ArXiv, abs/2402.01364*.

[2] Wen, S., & Itti, L. (2018). Overcoming catastrophic forgetting problem by weight consolidation and long-term memory. *arXiv preprint arXiv:1805.07441*.

[3] Rebuffi, S., Kolesnikov, A., Sperl, G., & Lampert, C.H. (2016). iCaRL: Incremental Classifier and Representation Learning. *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 5533-5542.

[4] Li, Z., & Hoiem, D. (2016). Learning without Forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 40*, 2935-2947.

[5] Shi, H., Xu, Z., Wang, H., Qin, W., Wang, W., Wang, Y., ... & Wang, H. (2024). Continual learning of large language models: A comprehensive survey. *ACM Computing Surveys*.

[6] Mallya, A., & Lazebnik, S. (2018). Packnet: Adding multiple tasks to a single network by iterative pruning. In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition* (pp. 7765-7773).

[7] Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H., & Ranzato, M. A. (2019). On tiny episodic memories in continual learning. *arXiv preprint arXiv:1902.10486*.

[8] Chen, W., Zhou, Y., Du, N., Huang, Y., Laudon, J., Chen, Z., & Cui, C. (2023, July). Lifelong language pretraining with distribution-specialized experts. In *International Conference on Machine Learning* (pp. 5383-5395). PMLR.

[9] Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, *23*(120), 1-39.

<br>

## Appendix

### 什么是Fisher信息矩阵？

Fisher 信息矩阵衡量模型输出分布对参数变化的敏感性，是连接统计推断、信息几何与神经网络防遗忘（EWC）的关键工具。具体来说，假设模型输出的概率分布为 $p(x|\theta)$，其中 $\theta$ 是模型参数（比如神经网络的权重）。我们希望知道：

> 如果我稍微改变参数 $\theta$，输出分布 $p(x|\theta)$ 会变化多少？

- **变化大 → 参数重要**（说明模型对该参数很敏感）；
- **变化小 → 参数可动性高**（不会影响太多）。

Fisher 信息矩阵正是用来**量化“模型输出对参数变化的敏感度”**的。在概率模型 $p(x|\theta)$ 下，Fisher 信息矩阵定义为：
$$
F(\theta) = 
\mathbb{E}_{x\sim p(x|\theta)}
\left[
\big(\nabla_\theta \log p(x|\theta)\big)
\big(\nabla_\theta \log p(x|\theta)\big)^\top
\right]
$$
这是一个 $d \times d$ 的对称半正定矩阵（$d$=参数维度）。

- 每个元素 $F_{ij}$ 表示参数 $\theta_i$ 与 $\theta_j$ 的协方敏感性；
- 对角线项 $F_{ii}$ 表示参数 $\theta_i$ 的“**重要程度**”。

如果更加直观地去看每个位置的定义：

- $\log p(x|\theta)$：代表对数似然（log-likelihood）；
- $\nabla_\theta \log p(x|\theta)$：表示改变参数后，对预测概率的影响方向；
- 然后对它平方求期望，就是衡量**平均敏感度**。

<br>

### 什么时候MoE的专家会被事先决定好？

工程上**可以**给出“软约束/先验”，让某些专家**更可能**学到特定数据的模式，但仍非硬编码语义：

1. **元信息引导路由**：给路由器输入语言/域/工具 ID（或把它拼进 token 表征），相当于给定一个**路由先验**。
2. **哈希/规则路由**：按哈希把 token 分片给专家，形成**硬划分**的“数据子分布”，再在其上训练。
3. **阶段式注入新专家**（Lifelong MoE）：新阶段只训练新专家 + 少量共享层，旧专家冻结；虽然不写着“这是医疗专家”，但因为只在医疗数据上更新，**自然成了“偏医疗”的专家**。
4. **专家先训再冻**：先用某个子集小调某些专家，再合回来训练其他数据（半监督的“定向专化”）。

**一个直观例子**

- 两次独立训练同一 MoE：都能学出“更擅长数字/括号/换行/代码关键字”的专家，但**编号不同**；Layer-5 的“数字专家”可能在另一跑里变成 Layer-6 的“数字专家”。
- 在 Lifelong MoE 场景，对“新语言批次”只开新专家并训练它们、冻结旧专家；训练后你会观察到“新语言 token 更常路由到新专家”，这是一种**受流程影响的专化**，但仍由损失和路由共同决定。

### Load Balancing 的做法

#### 1. 噪声路由（Noisy Top-k）

在计算路由打分时加入随机噪声：
$$
\tilde s_e = w_e^\top h + b_e + \sigma \epsilon_e
$$
噪声让“次优专家”偶尔被选中，从而避免塌陷。训练期使用较大的噪声鼓励探索，推理时去掉噪声以保证稳定。噪声可以遵从高斯分布或者Gumbel分布。推理时取消噪声（有点反向dropout的意思）

####  2. Keep Top-k + 容量控制

每个 token 只派发给 **Top-k** 个专家；每个专家能接收的 token 数上限为
$$
C = \text{capacity factor} \times \frac{kN}{E}
$$
超过上限的 token 会被 **丢弃(drop)** 或 **改派(reroute)** 给下一候选专家。这样可以限制计算量并防止单个专家过载。

#### 3. 均衡损失

为了让不同专家被使用得更均匀，训练时通常也会在主任务损失之外加入一个**辅助项（Auxiliary Loss）**。这个项衡量各专家的“**负载（Load）**”和“**重要性（Importance）**”是否分布均匀：

- **负载（Load）**：每个专家实际接收到的 token 数。
- **重要性（Importance）**：路由器为每个专家分配的平均概率权重。

若有的专家负载很大，而其他几乎闲置，说明路由不均衡，需要惩罚这种偏斜。一种常见做法是最小化它们的方差或变异系数（Coefficient Variation, CV²）：
$$
\mathcal L_{\text{balance}}
= \mathrm{Var}(\text{Load}) + \mathrm{Var}(\text{Importance})
$$
（也可写成基于均值归一化后的平方差或变异系数的形式）最终的总体损失函数为：
$$
\mathcal L = \mathcal L_{\text{LM}} + \lambda_{\text{bal}}\,\mathcal L_{\text{balance}},
$$
其中 $\lambda_{\text{bal}}$ 是控制均衡项权重的系数。