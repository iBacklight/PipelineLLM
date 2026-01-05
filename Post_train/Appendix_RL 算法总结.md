# RL 算法总结

接下来我们重点观察集中RLHF的具体算法。在RLHF中，我们有三个关键角色：

- **策略模型**：我们要优化的LLM
- **参考模型**：SFT后的模型，用于KL约束
- **奖励模型**：提供主要奖励信号

## 1. PPO

我们还是要从PPO开始学习，因为PPO是首次被正式通过RLHF的方法提出的强化学习算法 [1]。

PPO的算法本身我们之前的介绍基本一致，其主要的区别大概有：

1. 奖励来源基于RM和KL散度并且是稀疏奖励（即只有当序列生成结束才会给予奖励）
2. PPO-ptx: 在模型的最终目标中加入预训练的梯度

接下来我们来看如何使用PPO进行RLHF流程。

### 1.1 LLM训练中一般的RL的范式

| RL组件         | 在LLM生成中的具体定义                                        |
| :------------- | :----------------------------------------------------------- |
| **状态 (s_t)** | 当前上下文：`[prompt, y_1, y_2, ..., y_{t-1}]`               |
| **动作 (a_t)** | 从词汇表中选择下一个token `y_t`                              |
| **策略 (π)**   | LLM的概率分布                                                |
| **环境**       | 文本生成环境（确定性的状态转移）<br />environment is a bandit environment which presents a random customer prompt and expects a response to the prompt. [1] |
| **奖励 (r_t)** | 稀疏奖励：大部分步骤为0，结束时获得总奖励：<br />it produces a reward determined by the reward model and ends the episode.[1] |



### 1.2 算法实现：目标与损失

我们在第4节中已知奖励函数模型，即要最大化(1)式。在InstructGPT（也就是首次提出RLHF概念的论文）里，作者把PPO的损失函数写成，并命名为PPO-ptx（pre-train mix）：
$$
\mathbb{E}_{(x,y)\sim D_{\pi}}\Big[ r_\theta(x,y)\;-\;\beta\,\log\frac{\pi^{\text{RL}}_\phi(y|x)}{\pi^{\text{SFT}}(y|x)}\Big]
\;+\;\gamma\,\mathbb{E}_{x\sim D_{\text{pretrain}}}\big[\log \pi^{\text{RL}}_\phi(x)\big].
$$

1. **第一项 $R_\theta(x,y)$**：RM（或混合程序化指标）给的**奖励**，推动策略朝“人更喜欢/任务更成功”的方向移动。
2. **第二项 $-\beta \log \frac{\pi^{\text{RL}}}{\pi^{\text{SFT}}}$**：**KL 惩罚到参考策略（SFT）**。把期望拿进去就是 $-\beta\,\mathbb{E}[\,\log \pi^{\text{RL}}-\log \pi^{\text{SFT}}\,]=-\beta\cdot \mathrm{KL}(\pi^{\text{RL}}\|\pi^{\text{SFT}})$。
3. **第三项 $+\gamma\,\mathbb{E}_{x\sim D_{\text{pretrain}}}[\log \pi^{\text{RL}}(x)]$**：这就是 **ptx**（pretraining mix）项——在**预训练分布**上做一点点**语言建模极大似然**（maximize log-likelihood）。它直接给策略网络一个“像预训练那样会写自然语言”的**辅助梯度**，抵消 RLHF 里常见的退化（困惑度上升、语法变怪、知识回忆变差等）。

> 所以这里的超参数$\beta$ 管理 **KL 收紧强度**，$\gamma$ 管理 **预训练梯度的注入强度**。InstructGPT 里“PPO-ptx”指 $\gamma>0$，而普通“PPO”就是 $\gamma=0$。

这里可能有两个细节需要注意：

1）与PPO算法的等价性：

我们在上一章节中重点学习PPO的两种形式——Clip和KL，他们在数学上是等价的。但是我们发现，为什么这个目标函数中既没有重要性采样的ratio也没有之前强调的优势advantage呢？实际上，InstructGPT在实现PPO时，也使用了广义优势估计（GAE）来计算优势函数，并且使用了重要性采样。但是，在目标函数的表述中，他们将其写成了上面的期望的形式，而将PPO算法中的重要性采样和优势函数隐含在了优化过程中。

也就是说，我们实际上也是把PPO的实际loss写成

**版本 A：PPO-KL**

$$
\boxed{
\mathcal{L}_{\text{total}}= L_{\text{policy}}^{\text{KL-embed}} +c_v\,L_V +c_{\text{ent}}\,L_{\text{ent}}+\gamma\,L_{\text{ptx}}
}
$$

**版本 B：PPO-Clip**

$$
\boxed{\mathcal{L}_{\text{total}}= L_{\text{policy}}^{\text{clip}} +c_v\,L_V +c_{\text{ent}}\,L_{\text{ent}} +\gamma\,L_{\text{ptx}}
}
$$

在构建L_KL或者L_clip的时候，也都同时用到了ratio和advantage。那为什么不直接像PPO原文那样写成包含这两项直观的期望形式呢？这个没有直接的答案，我猜测主要是表达重心不同，但是似乎也没有定论：

- 他们想强调 **KL-to-ref** 与 **PTX** 这两个结构性设计；
- PPO-clip / PPO-penalty 属于广为人知的“怎么做”，篇幅上可能简写成“最大化期望回报（含 KL 正则）”。

我感觉实际目标更像是把上面的“真目标”按时间步展开并加入折扣（当然这里忽略了ptx）：
$$
J(\theta)=\mathbb{E}\!\left[\sum_{t} \gamma^t 
\Big(r_t^{\theta} - \beta\,\text{KL}_t(\pi_\theta\|\pi_{\text{ref}})\Big)\right].
$$

2）RM的序列与GAE时间步的关系

在实践中，InstructGPT和相关工作采用了一种**稀疏奖励**的设置：即除了最后一步外，其余步的环境奖为 0。可以说这个奖励是针对整个序列的，而不是每个时间步的。因此，我们需要将这个序列级别的奖励拆解为每个时间步的奖励：
$$
r_t^{\text{env}}=\begin{cases}
0,& t<T\\
R_{\text{RM}}(x,y_{1:T})-\alpha_{\text{len}}\cdot T,& t=T
\end{cases}
$$
如果遵循之前4.5的奖励模型范式，就每步再减去 KL 罚：$r_t=r_t^{\text{env}}-\beta\,\text{KL}_t$。也就是说：

- **$t<T$** 的步没有环境奖励（RM 只给整条序列一个分数），所以只记 **KL 负奖励**（稳定策略别跑远）。这一阶段也被称为***Rollout 采样阶段***。
  - 从初始状态（prompt）开始
  - 使用当前策略（LLM）采样动作序列
  - 直到终止状态（EOS）
  - **不更新策略参数**
- **$t=T$** 的最后一步把**RM 的序列分**一次性灌进来；如果要惩罚过长输出，会有长度惩罚 $\alpha_{\text{len}}\cdot T$。
- 也可以把 $R_{\text{RM}}$ 均匀或按权重分摊到各步，但“末步灌分”是最常见实现。

注意：**生成时（t 从 1 到 T）不做梯度更新，更新发生在整条序列结束之后。**但在更新阶段，**t<T 的所有步都会参与 loss**（即使它们没有环境奖励），因为它们要么有 KL（若把 KL 写进奖励或作为单独项），要么通过 GAE 从末步的回报“回流”出非零优势；同时每步还有 value loss 和熵正则。

```python
# 虽然奖励只在最后一步，但优势会传播到所有步骤
sparse_rewards = [0, 0, 0, ..., total_reward]  # 长度T
advantages = compute_gae(sparse_rewards, values)  # 长度T

# 每个时间步都有对应的优势值，用于梯度计算
for t in range(T):
    ratio = exp(new_log_probs[t] - old_log_probs[t])
    surr1 = ratio * advantages[t]    # 每个token都贡献梯度
    surr2 = clip(ratio, ...) * advantages[t]
```

### 1.3 核心代码实现

见代码库

### 1.4 PPO的局限

那么PPO的训练有哪些局限性亟待解决呢？

1. **价值函数训练的困难与不稳定性**
   - 在PPO中，价值函数用于估计状态的好坏，从而计算优势函数。然而，在文本生成任务中，“状态”是不断增长的序列，其价值非常难以准确估计。
   - 训练一个不稳定的价值函数会向策略模型注入噪声，成为整个系统的一个主要故障点。
2. **极高的计算和内存复杂度**
   - PPO需要同时维护和训练四个模型：**策略模型、价值模型、参考模型**和**奖励模型**。
   - 这导致了巨大的计算负担和内存开销，尤其是在模型参数达到数十亿甚至数百亿级别时。
3. **算法工程的复杂性**
   - 协调四个模型的训练流程、超参数调优（如价值函数学习率、优势估计的GAE参数等）非常复杂，使得重现和调试PPO训练变得异常困难。
4. **奖励**
   - 模型性能取决于标注奖励模型的人员的上限。
   - 奖励函数由人类反馈定义，可能具有高噪声和非平滑性，容易导致训练过程不稳定。
   - 除此之外，奖励函数的稀疏性可能会导致长期信用分配困难。这也很好理解，对于生成的长序列（或者思考过过程），无法判定每一步的生成的好坏；KL散度也只能证明是否偏离于基座，无法代表token是否合理。

<br>

### 5.2 GRPO

Group Relative Policy Optimization (GRPO)[3-4] 算法是由Deepseek提出的，它出发点是为了解决PPO算法的困境：**我们能否找到一个更简单、更直接的方式来利用相对偏好信息，从而完全避免训练一个显式的价值函数？**

#### 5.2.1 GRPO的原理

GRPO的原理建立在以下几个关键思想上：

- **从绝对奖励到相对优势**
  GRPO认为，我们并不需要知道一个回答的绝对价值是多少，我们只需要知道**在一个小群体里，哪个回答比另一个更好**就够了（相对价值）。这更符合我们拥有“偏好对”数据的本质。这样不但减小了计算量，也减少了方差。
- **群体作为基线**
  对于一个给定的提示，GRPO让当前的策略模型生成一个**群体（Group）** 的回答（例如，4-8个不同的回答）。这个群体内部就形成了一个微型的“质量分布”。通过计算这个群体内的**平均奖励**，我们得到了一个动态的、与当前策略相关的**基线**。
- **隐式优势估计**
  一个回答的优势（Advantage）被定义为它的奖励**超过群体平均奖励的量**。即：
  `优势(回答_i) = 奖励(回答_i) - 平均奖励(群体)`
  这个优势分数直观地反映了：“在这个上下文中，这个回答比模型自己通常生成的回答好多少？”

我认为其最主要的贡献就是提出了一种无需critic model就可以估算优势的方法。

#### 5.2.2 算法实现

我们来走一遍GRPO的流程，看看它是怎么实现无需critic model就可以估算优势的。但是在GRPO中，仍然需要一个RM，当然这个RM可能会有所不同的定义，我们后面再仔细看。

我们先给定一个 prompt $x$，从当前策略 $\pi_\theta$ **一次采样 K 条**完整输出 $\{y^{(i)}\}_{i=1}^K$， 用 RM 给每条完整输出一个**序列级奖励** $R^{(i)}=R(x,y^{(i)})$，并把这K条输出标**记为一个组**。接下来，我们对**同一组**里做**相对化**，把**绝对分** $R^{(i)}$ 变成**优势权重** $A^{(i)}$​：
$$
A^{(i)} \;=\tilde R^{(i)}=\frac{R^{(i)}-\mu_R}{\sigma_R+\epsilon},\quad 
\mu_R=\tfrac1K\sum_i R^{(i)},\ \ \sigma_R=\mathrm{Std}(\{R^{(i)}\})
$$
从论文 [4]给出的图片中，我们可以更加直观的看到它是怎么实现的：

![GRPO_comparsion](/home/awpc/studies/LLMs/Post_train/pics/GRPO_comp.png)

*Fig.1 GRPO 的分组打分策略，图片来自论文[4]*

我们可以看到，在对同一个prompt作多次生成采样后，我们得到了一组输出。然后我们根据RM计算每一个输出的rewrad。在计算优势Advantage的时候，由于没有Value/Critic Model，GRPO在原文中适用了组内均值基线（centered reward）
$$
A^{(i)} \;=\; \tilde R^{(i)} - \frac{1}{K}\sum_{j=1}^K \tilde R^{(j)}.
$$
也就是对第i个回答的优势就是该回答的总得分减去组内所有回答得分的均值后再做标准化：$A^{(i)} \leftarrow A^{(i)}/(\mathrm{Std}(\tilde R)+\epsilon)$ ）。而这个计算出的优势$A^{(i)}$将会同时赋给第i个完整输出的每一个token。这里我认为它是传递了一种新的思想：与其只给完整输出序列的最后一个token奖励，再通过链式的方法传播优势，不如我直接**将整个输出序列当作一个联合动作**，为这个大的动作赋予一个奖励和优势，以避免回流的不稳定性；另外这样做同时还减少了一个critic model的训练，大大降低了训练成本。注意：这只是一种思想，在 GRPO里，**策略依然是按 token 自回归的**。只不过优势通常是按照样本序列分配的。

同时KL散度也不被显式地加入奖励中，而是被加如后面的整体目标。直到了这些我们就可以列出GRPO的总目标函数：
$$
\boxed{
J_{\text{GRPO}}(\theta)
=\mathbb{E}_{\,x\sim\mathcal D,\ \{y^{(i)}\}_{i=1}^K\sim \pi_{\text{old}}(\cdot|x)}
\Bigg[
\frac{1}{K}\sum_{i=1}^{K}\frac{1}{T_i}\sum_{t=1}^{T_i}
\Big(
\underbrace{\min\big(\rho_{i,t}\,\hat A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\,\hat A_{i,t}\big)}_{\text{PPO-clip 形式}}
\ -\ \beta\,\underbrace{D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot|s^{(i)}_t)\,\|\,\pi_{\mathrm{ref}}(\cdot|s^{(i)}_t)\big)}_{\text{KL 到 SFT/ref}}
\Big)
\Bigg]
}
$$
这里的KL散度定义为：令 $r(o)=\frac{\pi_{\text{ref}}(o\mid q)}{\pi_\theta(o\mid q)}$。则
$$
\mathbb{D}_{\mathrm{KL}}\!\left(\pi_\theta\Vert \pi_{\text{ref}}\right)
= \mathbb{E}_{o\sim \pi_\theta}\big[\log \tfrac{\pi_\theta(o\mid q)}{\pi_{\text{ref}}(o\mid q)}\big]
= \mathbb{E}_{o\sim \pi_\theta}\big[r(o)-\log r(o)-1\big].
$$
推导一步步来看：
$$
\begin{aligned}
\mathbb{E}_{\pi_\theta}[r] 
&= \sum_o \pi_\theta(o)\frac{\pi_{\text{ref}}(o)}{\pi_\theta(o)} 
= \sum_o \pi_{\text{ref}}(o)=1,\\
\mathbb{E}_{\pi_\theta}[-\log r] 
&= \mathbb{E}_{\pi_\theta}\big[\log \pi_\theta-\log \pi_{\text{ref}}\big]
= \mathbb{D}_{\mathrm{KL}}(\pi_\theta\Vert \pi_{\text{ref}}).
\end{aligned}
$$
把这两项合起来就有
$$
\mathbb{E}_{\pi_\theta}[r-\log r-1]
= 1+\mathbb{D}_{\mathrm{KL}}(\pi_\theta\Vert \pi_{\text{ref}})-1
= \mathbb{D}_{\mathrm{KL}}(\pi_\theta\Vert \pi_{\text{ref}}).
$$
**为什么有 “−1”？** 因为 $\mathbb{E}_{\pi_\theta}[r]=1$，要让这个改写和原始 KL 完全等价，必须减去这一个常数 1 才不会引入偏移；同时也保证当 $r=1$（两分布相同）时，项 $r-\log r-1$ 取 0。该函数 $f(r)=r-\log r-1$ 也是一个经典的 **Itakura–Saito / f-divergence** 生成函数，满足 $f(r)\ge 0$ 且 $f(1)=0$。

实用层面中，单个采样动作就能估计 KL 惩罚，且对 $\log \pi_\theta$ 的梯度是 $1-r$，数值更稳定，不需要显式地再去算 $\log \pi_{\text{ref}}$ 的梯度。

损失函数则是最小化目标函数的负值
$$
\boxed{
\mathcal L_{\text{GRPO}}(\theta) = -\,J_{\text{GRPO}}(\theta)
}
$$
可以看到，GRPO不但引入了PPO的经典Clip形式，更是同时加入了KL正则化，使得模型能更加稳定的训练。除此之外，为了防止熵坍缩问题（Entropy Collapse，指在RL的训练过程中，策略熵快速下降，导致策略过早第收敛到一个局部最优解，策略概率分布此时看起来可能会比较尖锐，这会使得GRPO采样的组内的完整输出看起来差不多，优势函数也很相近没有差异），我们依然需要像PPO那样添加熵bonus项。

以上各符号与我们之前的一致：

- 组采样：对同一 prompt $x$ 采 $K$ 条完成 $y^{(i)}_{1:T_i}$（第 $i$ 条长度为 $T_i$-seq len）。

- 状态/动作：$s^{(i)}_t=[x, y^{(i)}_{1:t-1}]$，$a^{(i)}_t = y^{(i)}_t$。

- **比率**（用旧策略做重要性采样）：
  $$
  \rho_{i,t}\;=\;\frac{\pi_\theta(a^{(i)}_t\,|\,s^{(i)}_t)}{\pi_{\text{old}}(a^{(i)}_t\,|\,s^{(i)}_t)}.
  $$

这里插一嘴，这里的重要性采样依然是token-level的。这是因为虽然优势按照序列平均划分，但是序列的概率还是按 token 因式分解
$$
\pi_\theta(y_{1:T}\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid s_t),\quad s_t=[x,y_{<t}]
$$
因此用旧策略 $\pi_{\text{old}}$ 采样、用新策略 $\pi_\theta$ 评估时的**重要性采样比率**也随之分解。但是它不会去做对数和-指数等价计算，而是每一个token被这样采样后直接进行clip，最后求平均。这一点似乎时有风险的，我们在GSPO[5]算法的部分再详细讨论。



#### 5.2.3 奖励模型

GRPO奖励模型和PPO有所不同，在第4节中，我们概述了PPO的RM训练过程。GRPO刚提出时是为了解决数学问题，并没有提出新的训练范式，而是给出了两种不同的奖励源：

**1）ORM = Outcome Reward Model**

- 给整条完成序列一个**终端分**（对“最终答案/结果”的好坏打分，类似PPO，但是赋予整个序列的每个token）。
- 在 GRPO 里：对同一 prompt 一次采 K 条完成，得到 $\{R^{(i)}_{\text{ORM}}\}_{i=1..K}$。
  - 典型做法：长度归一或加长度惩罚，得到 $\tilde R^{(i)}$。
  - 组内相对化成优势（任选其一）：均值基线：$A^{(i)}=\tilde R^{(i)}-\frac1K\sum_j \tilde R^{(j)}$（可再除以组内 std）。
  - 每个样本的 **$A^{(i)}$** 通常**整条序列共享**，乘到该样本所有 token 的 $\log\pi_\theta$（再配 KL-to-ref）。

**2）PRM = Process Reward Model**

- 给**中间步骤**打分（如每一个推理步、关键中间结论、工具调用的正确性等）。这在解决数学问题上非常重要，相当于过程分。
- 在 GRPO 里有两种用法：
  1. **步级优势**（更细粒化）：把序列切成步骤片段（用分隔符或对齐标注），得到每步 $r_t^{\text{PRM}}$，再做折扣或者累计
     $\Rightarrow A_t^{(i)}$ 用在对应 token 片段；优点是信用分配好，训练更稳。
  2. **样本级汇总**（更简洁）：把各步分数加权平均成一个标量 $R^{(i)}_{\text{PRM-agg}}$，与 ORM 一样组内相对化得到 $A^{(i)}$。

但是原文并没有提到如何整合这两种模型，有些[最新的工作](https://arxiv.org/pdf/2508.05170)可能提到了具体的方法论，感兴趣可以了解。

然而在R1问世的时候，团队将这两种模型替换成了：

- **Accuracy reward**：能程序化判定的正确性（数学题答案匹配、代码通过单测、工具调用是否命中等），直接给 0/1 或分数。
- **Format reward**：检查输出是否遵守指定格式/结构（如必须含“思考/最终答案”两段、JSON 合法率等）。

原文中声称的好处有：

- 减少 reward hacking、分布外误判。
- 与 GRPO 天然契合：一次采 K 条，按“是否答对/格式合规”的**相对**好坏分配优势，稳定且简单。
- 适合可自动判定的任务（Math/Code/Structured QA/Tool）。

>We find that the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process, and retraining the reward model needs additional training resources and it complicates the whole training pipeline. [3]



#### 5.2.4 GRPO的局限性

GRPO解决了多种PPO存在的局限性问题，但是仍然没有解决：

1. **僵硬且可能无效的约束**

   - **问题**：GRPO使用初始的SFT模型作为单一的、固定的锚点。随着训练的进行，策略模型不断学习新的、可能更优的知识，但这个锚点却从未更新。这可能导致约束过于严格，限制了策略模型的探索和能力提升，或者因为锚点过于陈旧而导致约束无效。
   - **类比**：就像学骑车时，辅助轮永远不拆掉，虽然不会摔倒，但也永远无法学会真正的平衡和速度。

2. **Token-level的分配导致**

   - 高方差（尤其长序列）：GRPO 常用的 surrogate 形如 $\sum_t \rho_t \nabla\log\pi_\theta(y_t|s_t)$（样本级一个 $A$ 乘整条 token）。这种**逐 token 的 $\rho_t$** 仍会在长序列里引入**巨大方差**：某些 token 的 $\pi_{\text{old}}$ 很小（或 $\pi_\theta$ 偏移），$\rho_t$ 就会异常大/小，噪声放大，训练抖动明显。长度越长，这种“局部极端比率”出现的概率越高。

   - 目标错配：样本级优势 × token 级比率，GRPO 通常用**样本级**优势 $A^{(i)}$（整条序列共用一个权重）来优化**逐 token**的对数似然，但用 **token-level IS** 去“校正分布”会导致**粒度错配**。因为我们想校正的是“整条样本在旧/新策略下的采样偏差”，但却在每个 token 上做了独立的 IS。

   - 长度长的token的权重会被稀释：举个例子，我们假设现在有一个prompt并且我采样了三组回答

     - **A**：2 个 token，奖励 $r_A=+1$
     - **B**：6 个 token，奖励 $r_B=+1$（啰嗦）
     - **C**：6 个 token，奖励 $r_C=-1$

     设组均值 $\bar r=\frac{1+1-1}{3}=\tfrac{1}{3}$，于是有：
      $A_A=A_B=+\tfrac{2}{3},\ A_C=-\tfrac{4}{3}$​（组优势）。对于GRPO：一句话的优势 $A_i$ **平均摊**给该句所有 token。

     - A：每 token 得 $+\tfrac{2}{3}/2$
     - B：每 token 得 $+\tfrac{2}{3}/6$（被**稀释**，因为每个它哦肯可能作用和A中一样，但是数值上优势小于每一个A中的token）
     - C：每 token 得 $-\tfrac{4}{3}/6$ （惩罚也被稀释）

   - 在MoE架构下，会导致路由不连续，微小改动造成大跳变，这也和高方差有关

<br>

### 5.3 GSPO

Group Sequence Policy Optimizatioin (GSPO) [5] 是由Qwen团队提出的，基于GRPO的优化的强化学习算法。在GSPO中，序列级别的训练正式被提出。

#### 5.3.1 GSPO原理

GRPO 常见做法是**样本级优势（整条序列共用一个 A）× token 级比率/裁剪**。这样会带来三类问题：

- **高方差**：某些 token 的旧/新策略概率差异极端，token 级重要性比率在长序列里噪声累积，易不稳甚至崩溃；裁剪又引入系统性偏差（步步叠加）。
- **粒度错配**：我们想奖励“整条答案更好”，却在每个 token 上独立做 IS/clip，导致样本级信号被 token 噪声扭曲。
- **工程脆弱**：温度/Top-p、过滤等造成的**序列级**分布偏差，用 token-IS 很难正确修正。
  GSPO 的核心改动是：**把比率、裁剪、奖励和优化全部提升到“序列级”**，并做**长度归一**，显著提升稳定性与效率，特别适合长序列与 MoE 训练。

GSPO 的核心改动是：**把比率、裁剪、奖励和优化全部提升到“序列级”**，并做**长度归一**，显著提升稳定性与效率，特别适合长序列与 MoE 训练。



#### 5.3.2 算法实现

我们依然给定 prompt $x$，模型从旧策略 $\pi_{\text{old}}$ 抽 $K$ 条完成 $y_i$（组采样）。此时GSPO的目标函数为：
$$
\boxed{
J_{\text{GSPO}}(\theta)
=\mathbb{E}_{x\sim\mathcal D,\; \{y_i\}_{i=1}^G\sim \pi_{\text{old}}(\cdot|x)}
\left[
\frac{1}{K}\sum_{i=1}^{K}
\min\big( s_i(\theta)\,\hat A_i,\ \text{clip}(s_i(\theta),\,1-\varepsilon,\,1+\varepsilon)\,\hat A_i \big)
\right].
}
$$
这里的组内优势也和GRPO一致：
$$
\boxed{
\hat A_i
= \frac{\,r(x,y_i) - \mathrm{mean}\big(\{r(x,y_j)\}_{j=1}^{G}\big)\,}
{\,\mathrm{std}\big(\{r(x,y_j)\}_{j=1}^{G}\big)+\epsilon\,}.
}
$$
所以这看上去好像和GRPO很相似。主要是这里的$s_i(\theta)$​, 也就是是**序列级重要性比率**（Sequential level IS）有所不同，它定义为：
$$
\boxed{
s_i(\theta)
=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\right)^{\tfrac{1}{|y_i|}}
=\exp\!\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})}\right).
}
$$
这表示，作者依然还是仿照PPO/GRPO的方法计算把序列级概率比值提升到 $1/|y_i|$ 次方。其中

- $|y_i|$ 表示第 $i$ 条完成序列的**长度（token 数）**。
- $\Big(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}\Big)^{\!\frac{1}{|y_i|}}$ 就是把**整条序列的概率比**做**按长度的几何平均**（length normalization）。

为了更加清楚GSPO和GRPO的区别，我们仔细再分析一下二者的区别：

- **GRPO（token-level IS/clip）**：对**每个 token**算比率并裁剪，然后再在时间维上求和/取均值。
- **GSPO（sequence-level IS/clip）**：先把整条序列的 log-likelihood 做**长度归一**后合成一个**序列级比率**，**在序列层**裁剪，一次性作用于整条完成。

我们给定组内样本 $i$ 的优势 $A_i$（整条序列共享，之前说过两个算法在这里是等价的），为了方便比较，我们列出GRPO的token 级比率：
$$
\rho_{i,t}=\frac{\pi_\theta(y^{(i)}_t\mid s^{(i)}_t)}{\pi_{\text{old}}(y^{(i)}_t\mid s^{(i)}_t)}.
$$
GRPO 的 surrogate目标（PPO-clip 的 token 版）：
$$
\boxed{
J_{\text{GRPO}}(\theta)
=\mathbb{E}_{\,x\sim\mathcal D,\ \{y^{(i)}\}_{i=1}^K\sim \pi_{\text{old}}(\cdot|x)}
\Bigg[
\frac{1}{K}\sum_{i=1}^{K}\frac{1}{T_i}\sum_{t=1}^{T_i}
\Big(
\underbrace{\min\big(\rho_{i,t}\,\hat A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\,\hat A_{i,t}\big)}_{\text{PPO-clip 形式}}
\ -\ \beta\,\underbrace{D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot|s^{(i)}_t)\,\|\,\pi_{\mathrm{ref}}(\cdot|s^{(i)}_t)\big)}_{\text{KL 到 SFT/ref}}
\Big)
\Bigg]
}
$$
可以看出，GRPO的IS是先在单个生成的完整序列中的**每一个token**上作surrogate的clip和运算（纵然adv是序列级别的token全都相等）。而GSPO则不同，在其目标函数中，并没有token级别的运算，而是在一次计算完序列整体IS后，只在采样的K次维度上面作平均。我们举一个例子：

设优势 $A=1$, Clip裁减的超参$\varepsilon=0.2$, 当前prompt的回答只有两个token，第一个token的IS ratio $\rho_1=2.0$, 第二个token的IS ratio $\rho_2=0.4$。

- **GRPO**：
  clip 1 -> $\min(2,1.2)=1.2$, clip 2 -> $\min(0.4,0.8)=0.8$，求平均 $(1.2+0.8)/2=1.0$。

- **GSPO**：

  $s=\exp\big((\ln2+\ln0.4)/2\big)=\exp(-0.1116)\approx0.894$，
  $\min(0.894,\,\text{clip}(0.894,0.8,1.2))=0.894$。

这么做的好处是：

- **方差**：GRPO 的 token 比率在长序列易出现极端值，方差大；GSPO 用“几何平均的序列比率”更稳。
- **对齐**：奖励/优势多是**序列级**；GSPO 的序列级比率与之粒度匹配，GRPO 容易被某些 token 的极端比率扭曲。
- **调参**：GSPO 用 length-normalized 比率，$\varepsilon$ 与 `target_kl` 可跨长度复用；GRPO 的裁剪量级常与长度、token 波动强相关。

#### 5.3.3 GSPO缺点

长样本中token会被弱稀释：GSPO 的样本损失（正统写法）：
$$
\mathcal{L}_i^{\text{GSPO}}
= -\,A_i\cdot \frac{1}{T_i}\sum_{t=1}^{T_i}\underbrace{\text{clip}(\rho_{it})}_{\text{token级IS}}
\log\pi_\theta(y_{it}\mid x,y_{i,<t})
$$

- 与 **GRPO** 不同：GSPO **不**做 $A_i/T_i$ 的“优势均摊”，只对**token损失取平均**；因此**整条样本的总权重是 $A_i$**，**与长度无关**。
- 但平均意味着：如果长答里**关键token占比小**，它们的梯度会被大量“无关/赘述token”的平均稀释一部分（弱稀释）。

**举个例子**

- 两条同样优势 $A_i=1$：短答 $T=2$，关键token占 2/2；长答 $T=10$，关键token占 2/10。假设关键token的 $\text{clip}(\rho_{it})\log\pi_t$ 绝对值更大，其他 token 很小。
- **短答**：平均 = (大+大)/2 → “大”。
- **长答**：平均 = (大+大+8个小)/10 → 被“8个小”拉低。
  => 样本总权重相同（都是乘 $A_i$），但**关键token的相对影响**在长答里更小了一些——这就是“弱稀释”。

<br>

### 5.4 DAPO

Dynamic sAmpling Policy Optimization (DAPO) 是由字节seed和清华AIR提出的强化学习算法，解决了很多未被GRPO提出的训练问题（笔者觉得GSPO和DAPO都以GRPO为改进对象，可见GRPO对业界后训练test-time scaling的影响还是很大的）。我们来具体看一下有哪些改进。

#### 5.4.1 DAPO原理

 DAPO的作者在原文中提到，在大语言模型的 长链式推理（Long *Chain-of-Thought, Long-CoT）强化学习阶段，传统的 RL 方法（如 PPO、GRPO）往往出现以下问题：

* 训练不稳定：奖励信号在长推理序列中传递困难，噪声大；

* 模式坍缩（entropy collapse）：模型倾向于生成高度相似、低多样性的回答；

* 低效采样：长序列训练代价高，样本利用率低。

作者提出了提出了四项关键技术，使强化学习在长链式思维（Long-CoT）场景下重新焕发光彩：

* Clip-Higher：提升生成多样性，防止熵坍缩；

* Dynamic Sampling（动态采样）：提高训练效率与稳定性；

* Token-Level Policy Gradient Loss（基于 Token 的策略梯度损失）：在长链式推理场景中至关重要；

* Overlong Reward Shaping（超长奖励塑形）：减少奖励噪声，稳定训练过程。

我们将在Day 4的拓展节中，详细介绍长思维链(chain of thought, COT)。现在我们只学习DAPO的思路，以及它如何优化GRPO的缺陷。

#### 5.4.2 算法原理

设对每个问题 $x$ 由旧策略 $\pi_{\theta_{\text{old}}}$ 采样 $K$ 条输出 $y_i$，第 $t$ 个 token 的重要性比率设对每个问题prompt $x$ 由旧策略 $\pi_{\theta_{\text{old}}}$ 采样 $K$ 条输出 $y_i$，第 $t$ 个 token 的重要性比率
$$
r_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}\mid x,y_{i,<t})},
$$
组相对优势 $\hat A_{i,t}$ 由组内奖励标准化得到（与GRPO的风格一致）。DAPO 的目标是：
$$
J_{\text{DAPO}}(\theta)
=\mathbb{E}\Big[\frac{1}{\sum_i |y_i|}\sum_{i=1}^K\sum_{t=1}^{|y_i|}
\min\big(r_{i,t}(\theta)\hat A_{i,t},\ 
\mathrm{clip}\big(r_{i,t}(\theta),\ 1-\varepsilon_{\text{low}},\ 1+\varepsilon_{\text{high}}\big)\hat A_{i,t}\big)\Big],\\
s.t.~ 0 < \big|\{\,y_i \mid \mathrm{is\_equivalent}(y_i, y)\,\}\big| < K,
$$
$\varepsilon_{\text{low}},\varepsilon_{\text{high}}$ 分别是下/上界，论文实用取值如 $\varepsilon_{\text{low}}=0.2,\ \varepsilon_{\text{high}}=0.28$。而该动态限制条件代表函数只在“**该题组并非全对也并非全错**”时参与更新，否则优势全为常数，等于无效梯度。与此同时，作者也在目标中移除了KL散度项，原文中提到因为KL散度原本是为了束缚模型更新不能太多偏离于冻结的ref模型而加入的，但是在训练COT能力的时候，模型的分布可能会与基座模型大相径庭，因此该KL约束是不必要的。

该算法的核心卖点基于以下四个方面：

1）**Clip-Higher**：抬高上界来对抗“熵坍缩”

- 出发点：PPO/GRPO 在做clip时，超参数$\epsilon=0.2$​且上下截断范围相等的（0.8-1.2）。我们来看两个例子：

  - 一个 token 的原始概率是 $\pi_{\text{old}}(o_i|q) = 0.01$（低概率 token，代表**探索 token**）；
  - 另一个是 $\pi_{\text{old}}(o_i|q) = 0.9$（高概率 token，代表**利用 token**）。

  按照 PPO 的裁剪上限公式：
  $$
  \pi_{\theta}(o_i|q) \le \pi_{\text{old}}(o_i|q) \cdot (1+\varepsilon),
  $$
  上限为：
  $$
  0.01 \times 1.2 = 0.012,\quad 0.9 \times 1.2 = 1.08.
  $$
  可以看到：

  - 对高概率的“利用”token（0.9）来说，即使乘上 1.2，也几乎没限制，它甚至能进一步增大概率。
  - 但对于低概率的“探索”token（0.01）来说，最多只能增加到 0.012 —— 几乎没变化。

  这会导致**上限裁剪实际上限制了低概率 token（探索行为）的上升空间**，而对高概率 token 几乎没有影响。因此在训练中，模型更容易不断强化原本高概率的“确定答案”，而难以尝试新的思路或表达。久而久之会导致**样本同质化、探索受限、熵坍缩（熵快速下降）**。

- 做法：**解偶 $\varepsilon_{\text{low}},\varepsilon_{\text{high}}$**，**只增大上界**（例如 0.28），为“低概率但有正优势的探索 token”保留更大上升空间；下界维持不变，避免把不良 token 压到接近 0 造成采样空间塌缩。实验显示熵下降得到遏制（熵可以收敛在0.4附近而不是迅速降到0）、AIME 精度上升。

2）**Dynamic Sampling**：过滤“全对/全错”的采样，用来稳定有效梯度

  - **出发点**：GRPO 组内优势标准化做法是：

$$
  \hat{A}_{i,t}
  = \frac{R_i - \mathrm{mean}\big(\{R_i\}_{i=1}^K\big)}{\mathrm{std}\big(\{R_i\}_{i=1}^K\big)},
$$

  其中 $R_i = R(y_i, y)$ 即上面定义的奖励。那么如果该题 $K$ 条输出全对或全错，减去平均值会使得**优势全为 0**，从而使得梯度对该题无贡献；随着训练变强，“全对”比例持续上升，等价于**有效 batch 变小**、方差变大、学习不稳。

- 做法：在一个**动态缓冲**里持续采样并**滤掉 所有acc=1 或 0 的题**，直到缓冲区凑满“有有效梯度”的题再做一次更新（over-sample to fill）。这样每步都维持“有效样本量”与梯度信噪比，提高效率与稳定性。*注意，该做法并没有影响模型的效率，因为模型的生成效率主要取决于长尾samples的生成速度*。

3）**Token-Level Policy Gradient Loss**：抑制“长样本稀释效应”

- 出发点：这一点是为优化长思维链提出的。传统**token-level**的做法是“样本内先按 token 平均、再样本间平均”。这样在计算 loss 时，每个样本（response）被赋予了相同的权重， 所以在长样本（token 更多）的情况下，这些 token 各自对总体 loss 的贡献反而变得更小（长token的作用被稀释）。而这主要会造成两点负面影响：1）对于高质量的长样本，这种做法会阻碍模型学习推理能力。2）长输出会把自身坏模式（复读、胡言）的惩罚**被平均稀释**，导致**长度非健康增长**、熵异常走高。

- 做法：直接在**token 粒度**聚合损失（ **Token-Level Aggregation**）：
  $$
  \mathcal{L}_{\text{token-level}} = 
  \frac{1}{\sum_i |y_i|}
  \sum_{i=1}^{K} \sum_{t=1}^{|y_i|} \ell_{i,t}
  $$
  区别在于：不再“先样本平均，再 token 平均”；而是 **直接对所有 token 平均**，所有输出的序列长度都被考虑在内。这让每个 token 的梯度都真实进账；如果某段长样本里出现“坏 token”，它的负面贡献不会被同样本内其他 token 稀释，从而抑制胡言/复读与无谓增长。

  这里也可以对比一下GSPO的序列级做法，它先对**每条响应**算一次优势/损失，再按样本做平均：
  $$
  \mathcal L_{\text{GSPO-tl}}=\frac{1}{K}\sum_{i=1}^K\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\ell_{i,t}\right)
  $$
  结果：每条样本权重相等，**长样本会把自身的 token 梯度除以 $|y_i|$**，单个 token 的影响被稀释（length bias）。而DAPO是最后直接对**全批次 token**平均，目的是每个 token 的权重一致，**消除了样本长度对单个 token 梯度的稀释**。

4） **Overlong Reward Shaping**：降低“截断样本”带来的奖赏噪声

- 出发点：长-CoT 训练常设最大生成步数 $L_{\max}$。**被截断**（truncated）的样本若一律按“答错（-1）”记奖赏，会把“本可答对但未写完”与“真的答错”混淆，**引入噪声**。

- 做法两步走：

  1. **Overlong Filtering**：先**屏蔽**被截断样本的损失回传，训练明显更稳。

  2. **Soft Overlong Punishment**：在 $L_{\max}$ 前设置一个奖励的**软惩罚区间**，
     $$
     R_{\text{length}}(y) = \begin{cases}
     0, & |y| \le L_{\max} - L_{\text{cache}}, \\[6pt]
     \dfrac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \le L_{\max}, \\[8pt]
     -1, & L_{\max} < |y|.\end{cases}
     $$


     长度越接近/超过上限，额外惩罚越大；超过上限直接-1。这既给训练传达“回复别太长”的信号，又避免把所有截断样本一刀切按照“答错”计算奖励。

#### 5.4.3 奖励（非模型）

原文中不再使用训练过的rewards model，而是在“可验证任务”（verifiable task）中使用**基于最终答案正确性的离散奖励**：
$$
R(\hat{y}, y) =
\begin{cases}
+1, & \text{if } \mathrm{is\_equivalent}(\hat{y}, y),\\[6pt]
-1, & \text{otherwise},
\end{cases}
$$
（原文中写道是为了解决奖励模型带来的reward hacking问题）。`is_equivalent(y^,y)` 是判定两者是否语义或数值等价的布尔函数。在此基础上还要加上在5.4.2- 4）提到的Soft Overlong Punishment $R_{\text{length}}(y)$，形成了最终的奖励函数。

#### 5.4.4 DAPO的局限性

DAPO作为在CoT长样本上有明显优化的算法，已知的局限如下：

1）**奖励稀疏／信号弱**
 DAPO 在很多任务里用的是二元结果奖励（正确 / 错误 +1 / −1），这种“终局奖励”信号非常稀疏。虽然 Overlong Reward Shaping、动态采样 等能缓解一部分截断噪声和无效样本问题，但在更加复杂或主观任务（非可验证任务）里，这样的奖励形式可能不够强，对训练收敛较慢。然而reward就是这样的，永远在重复着稀疏和reward hacking的trade off.

2）**缺乏细粒度反馈 / 局部归因能力弱**
虽然 DAPO 用 Token-Level Loss，但如果优势 $\hat A_{i,t}$ 在序列内几乎一致，就无法明确指出哪几段 token 是“坏”或重复的问题。换言之，DAPO 本身在“坏片段精确定罪”的能力上仍然有限。

3）**泛化 / 任务限制**
DAPO 在长链式思维 / 数学 /可验证任务上表现被验证较好，但对复杂的开放式生成、对话、人类偏好类任务上的效果仍待检验。


## 6. ARPO
> Incoming

## 7. SAPO
> Incoming

