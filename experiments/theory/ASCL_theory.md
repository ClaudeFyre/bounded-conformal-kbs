# ASCL 理论草稿（Stage 1）

> 状态标记：**[已证]** 证明完整；**[草稿]** 证明思路清楚、细节待补；**[猜想]** 未证明。
> 所有结论都以一个**固定的解码器**做反事实比较：同一个解码器，换不同的潜变量后验。独立训练的两个模型之间，还会叠加编码器/解码器训练动态的差异；这一差异理论不覆盖，必须在论文中明说（见第 5 节）。

## 0. 记号

- 可行域 $\mathcal M=[0,1]^d$；解码器均值 $m:\mathbb R^d\to[0,1]^p$，解码器方差 $v:\mathbb R^d\to\mathbb R_{\ge0}^p$。
- Gaussian 后验记作 $q^G$。泄漏为 $\delta=q^G(h\notin\mathcal M)$，参考后验为 $q^\star=q^G(\cdot\mid h\in\mathcal M)$，外溢部分为 $q^\circ=q^G(\cdot\mid h\notin\mathcal M)$。
- **混合恒等式**：$q^G=(1-\delta)\,q^\star+\delta\,q^\circ$。
- 对后验 $q$，定义：
  - 预测 $\hat y_q=\mathbb E_q[m(h)]$；
  - 样本内方差 $W_q=\tfrac1p\mathbb E_q[\mathbf 1^\top v(h)]$；
  - 样本间方差 $V_q=\tfrac1p\mathbb E_q\|m(h)-\hat y_q\|^2$；
  - 尺度 $\hat\sigma_q^2=W_q+V_q$（即代码中 $S\to\infty$ 的总方差），得分 $\mathcal S_q=\|y-\hat y_q\|/\hat\sigma_q$。
- 简记 $m^\star=\hat y_{q^\star}$，$m^\circ=\hat y_{q^\circ}$，解码器位移 $\Delta=\hat y_{q^G}-m^\star$。

## 1. 宽度的精确分解 [已证]

在一个 Mondrian 组 $g$ 内，区间半径为 $\hat q_g\hat\sigma(x)$，所以平均宽度 $\bar W=2\hat q_g\bar\sigma_g$，其中 $\bar\sigma_g$ 是组内 $\hat\sigma$ 的平均。因此对任意两个模型 A、B：
$$\log\frac{\bar W_A}{\bar W_B}=\underbrace{\log\frac{\hat q_A}{\hat q_B}}_{\log R_q}+\underbrace{\log\frac{\bar\sigma_A}{\bar\sigma_B}}_{\log R_\sigma}.$$
这是恒等式，`scripts/decompose.py` 就按它计算。

## 2. 泄漏如何进入预测与尺度

**命题 A（预测位移）[已证]**：$\Delta=\delta\,(m^\circ-m^\star)$。
*证*：对混合分布取期望即得。

**命题 B（尺度通道，精确式）[已证]**：
$$W_{q^G}-W_{q^\star}=\delta\,(W_{q^\circ}-W_{q^\star}),\qquad V_{q^G}-V_{q^\star}=\delta\,(V_{q^\circ}-V_{q^\star})+\frac{1-\delta}{\delta}\cdot\frac{\|\Delta\|^2}{p}.$$
*证*：第一式由期望的线性性得到。第二式用混合分布的全方差公式：
$\mathrm{Var}_{q^G}=(1-\delta)\mathrm{Var}_{q^\star}+\delta\mathrm{Var}_{q^\circ}+\delta(1-\delta)\|m^\circ-m^\star\|^2$，再代入命题 A。

**解读**：
- 泄漏抬高尺度的主导项是 $\frac{1-\delta}{\delta}\|\Delta\|^2$，也就是"外溢样本把解码器输出拉开"。
- 泄漏小而位移不小时，这一项反而大。这对应数据中 TN-hetero 解码器下尺度通道占主导的现象（五个数据集中 H1 都成立）。

## 3. 分位数通道何时为负 [已证]

记相对尺度膨胀 $\kappa=\hat\sigma_{q^G}^2/\hat\sigma_{q^\star}^2-1$。对单元 $(x,y)$，令 $r^\star=y-m^\star$，则
$$\frac{\mathcal S_{q^G}}{\mathcal S_{q^\star}}=\frac{\|r^\star-\Delta\|}{\|r^\star\|}\cdot\frac{1}{\sqrt{1+\kappa}}\;\le\;\frac{1+\|\Delta\|/\|r^\star\|}{\sqrt{1+\kappa}}.$$

**命题 C（分位数收缩）[已证]**：若组内每个单元都满足
$$\kappa\;\ge\;2\,\frac{\|\Delta\|}{\|r^\star\|}+\Big(\frac{\|\Delta\|}{\|r^\star\|}\Big)^2,$$
则组内每个单元都有 $\mathcal S_{q^G}\le\mathcal S_{q^\star}$，从而经验分位数 $\hat q^G\le\hat q^\star$。
*证*：条件等价于 $(1+\|\Delta\|/\|r^\star\|)^2\le1+\kappa$；得分逐点不增，所以顺序统计量不增。

**解读**：
- 归一化尺度吸收的泄漏效应，比泄漏抬高的残差更多时，分位数反而下降。这解释了数据中五个数据集里有四个出现 $q^G<q^\star$，而旧定理 5 预测的是相反方向。
- 条件对每个单元都要求，太强；实际中只需要在分位数附近成立。**[草稿]**：把"每个单元"放宽到分位数临界集，可以沿用旧定理 5 的 $m$、$m_-$ 论证。

**推论（通道由什么决定）[草稿]**：同方差解码器的 $W_q$ 含一个全局常数底 $\sigma_0^2$，所以 $\kappa\approx\frac{(1-\delta)\|\Delta\|^2}{\delta p\,\sigma_0^2}$。当 $\sigma_0^2$ 相对样本间方差很大时，$\kappa$ 很小，命题 C 的条件不成立，泄漏就只能体现在残差（分位数）上。这给出"是否归一化 / 解码器是否异方差决定走哪条通道"的机制。与数据对照：H2 在三个检验数据集中 2/3 成立，Syn-Extreme 是反例。要逐数据集测 $\sigma_0^2$ 与样本间方差的比值来核对。

## 4. 支撑–表达力权衡

有界分布族 $\mathcal Q$ 的表示代价定义为 $E_{\mathcal Q}(x)=\inf_{q\in\mathcal Q}\mathrm{KL}(q^\star\,\|\,q)$，取到下确界的分布记作 $q^B$。

**命题 D（接近参考后验的充分条件）[已证]**：令 $R=\sup_{h,h'\in\mathcal M}\|m(h)-m(h')\|\le\sqrt p$，则
$$\|\hat y_{q^B}-m^\star\|\le R\sqrt{E_{\mathcal Q}/2},\qquad |V_{q^B}-V_{q^\star}|\le \tfrac{2R^2}{p}\sqrt{E_{\mathcal Q}/2}.$$
所以只要
$$R\sqrt{E_{\mathcal Q}/2}\;<\;\delta\,\|m^\circ-m^\star\|,$$
有界分布族在预测上就比 Gaussian 更接近参考后验。
*证*：$\hat y_q$ 是有界函数的期望，差值不超过 $R\cdot\mathrm{TV}$，再用 Pinsker 不等式；样本间方差同理。Gaussian 一侧用命题 A。

**推论（泄漏门槛）[已证，作为充分条件]**：有界约束在预测上值得施加的充分条件是
$$\delta\;>\;M^\star(E_{\mathcal Q})=\frac{R\sqrt{E_{\mathcal Q}/2}}{\|m^\circ-m^\star\|}.$$
泄漏低（如 ResOpsUS 的 $\delta\approx0.008$）时门槛几乎不可能达到，除非 $E_{\mathcal Q}\approx0$；这与观测一致。

**局限（必须写进论文）**：
1. "更接近 $q^\star$"不等于"区间更窄"。只有当参考后验本身是高效的，这个充分条件才转化为效率，还要叠加第 2、3 节的通道。
2. 训练中分布族的参数由 ELBO 决定，并不是按 $\mathrm{KL}(q^\star\|q)$ 最小化，实际代价可能大于 $E_{\mathcal Q}$。
3. 截断 Gaussian 族关于 $q^\star$ 有 $E_{\mathcal Q}=0$，理论上在反事实意义下不会更差；但独立训练时它在 ResOpsUS 上差了 14.7%（p = 0.037）。所以训练动态本身不可忽略，这是现有理论解释不了的部分。

## 5. 待办（Stage 1 的剩余工作）

1. **反事实检验**（不重新训练，能直接检验第 2–4 节）：用已存的 Gaussian 检查点和同一个解码器，分别在 $q^G$、$q^\star$、以及对 $q^\star$ 做 KL 投影得到的 $q^B$（逐单元数值拟合）下，计算宽度和 $R_q$、$R_\sigma$。这样可以把"理论覆盖的部分"和"训练动态造成的部分"分开。
2. 逐数据集估计 $E_{\mathcal Q}$ 和 $M^\star$，看门槛能否预测各数据集的符号（已知五个数据集的结果，只能作为发现，不能作为确认）。
3. 把命题 C 的逐点条件放宽到分位数临界集。
4. **[猜想]** 对独立训练的模型，给出与训练动态有关的界（比如以 ELBO 差为代价项）。这可能是理论中最难的部分。
