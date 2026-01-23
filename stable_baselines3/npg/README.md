# NPG 与 TRPO 实现差异对比

本 NPG 实现专门针对 "detach" 基线版本（在某些代码库中被称为 `mlp_true` 配置）进行了对齐，这与 SB3-Contrib 或 SpinningUp 等库中的标准 TRPO 或 vanilla NPG 实现有显著差异。

以下是核心架构和算法上的关键差异总结。

## 1. 优化机制 (核心差异)

- **标准 TRPO**: 使用 **共轭梯度 (Conjugate Gradient)** 寻找搜索方向，随后进行 **线性搜索 (Line Search / Backtracking)**，以确保严格满足 KL 散度约束并保证目标函数真正提升。它通常直接根据 $\sqrt{2 \delta / (g^T H^{-1} g)}$ 计算步长。
- **本 NPG 实现**:
  - 同样使用共轭梯度计算自然梯度方向 $H^{-1}g$。
  - **无线性搜索**: 不显式搜索步长，而是将计算出的自然梯度方向直接写入参数的 `.grad` 属性中。
  - **SGD 更新**: 使用标准的 **SGD 优化器** (momentum=0) 执行单步更新。
  - **自适应学习率**: 步长完全由 `lr_pi` 控制，而 `lr_pi` 会根据每次更新后观测到的实际 KL 散度进行 **自适应调整** (`use_kl_adaptive_lr`)。逻辑如下（其中 `target_kl` 默认为 0.008）：
    - 如果 $KL > 2.0 \times \text{target\_kl}$ (即 > 0.016)，减小学习率。
    - 如果 $KL < 0.5 \times \text{target\_kl}$ (即 < 0.004)，增大学习率。

## 2. 共轭梯度 (CG) 实现细节

- **SB3-Contrib**: 初始化搜索向量 $x$ 时会加入 **微小的随机高斯噪声**，以防止梯度过小时出现数值不稳定。
- **本 NPG 实现**: 将搜索向量 $x$ 初始化为 **全零向量**。这与 "detach" 参考实现对齐，并且在迭代步数较少（如 10-15 步）时能保证确定性，避免噪声干扰梯度的主要方向。

## 3. Fisher 矩阵 / KL 参考分布

- **标准 TRPO**: 通常计算 **旧策略 (old policy)** (固定) 与 **当前策略 (current policy)** (变量) 之间的 KL 散度来构造 Hessian 向量积 (HVP)。
- **本 NPG 实现**: 计算 **当前策略 (current policy)** 与 **其自身 (detached)** 之间的 KL 散度。
  - `kl_ref = "current"`
  - 这实际上是在计算当前参数点 $\theta_k$ 处的局部曲率 (Fisher 信息矩阵)。
  - 这种方式理论上更能保证 Fisher 矩阵是半正定的 (PSD)。

## 4. Critic (价值函数) 优化

- **标准 TRPO**: 常使用 L-BFGS 优化器，或者与 Actor 共享优化器结构。
- **本 NPG 实现**:
  - 使用 **完全独立的 Adam 优化器**。
  - **固定学习率**: Critic 的学习率 (`lr_v`) 是固定的，不随时间衰减，也不跟随 Actor 的自适应调整。
  - **梯度裁剪**: 对 Critic 应用显式的梯度裁剪 (`max_grad_norm_v`, 默认 5.0)。
  - **无 PopArt**: 与某些 "true" 基线不同，当前实现主要依赖 `VecNormalize` (环境装饰器) 来处理奖励缩放，而不是内部的 PopArt 价值归一化。

## 5. 动作分布与 Squash

- **标准 SB3**:
  - 使用 `DiagGaussianDistribution`，通常带有一个全局可学习的 `log_std` 参数 (与状态无关)。
  - 动作通常在采样 *后* 进行 clip，或者使用 `SquashedGaussian` (在 PPO/TRPO 中较少见)。
- **本 NPG 实现**:
  - **Action Squash**: 如果开启 `action_squash=True`，策略输出的是原始高斯样本，这些样本在与环境交互 *前* 会经过 `tanh` 函数 (类似 SAC，但在 NPG 更新计算 log_prob 时暂未包含 Jacobian 修正)。
  - **State-Dependent Std**: 支持 `state_dependent_std=True`，此时策略网络会输出 `mean` 和 `log_std` (输出维度翻倍)，使方差成为状态的函数。

## 6. 优势函数 (Advantage) 处理

- **标准做法**: 通常进行标准化 $(A - \mu) / (\sigma + \epsilon)$。
- **本 NPG 实现**:
  - 默认倾向于只做 **零均值化 (Zero-centering)**: $A = A - \text{mean}(A)$。
  - 虽然也支持标准归一化，但在对齐 "detach" 版本时，通常强调只减均值。
  - **可选开关（推荐）**:
    - `adv_subtract_mean`（默认 `True`）：控制是否做 \(A \leftarrow A - \mathrm{mean}(A)\)
    - `adv_divide_std`（默认 `False`）：控制是否做 \(A \leftarrow A / (\mathrm{std}(A) + \epsilon)\)

## 7. 超参数 (对齐 `mlp_true`)

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `grad_mode` | `npg` | 可切换为 `pg` 以禁用 CG |
| `post_grad` | `fisher_clip` | 根据 Fisher 模长对更新步进行归一化/裁剪 |
| `cg_damping` | `0.1` | 阻尼系数比通常设定 (0.01) 更大 |
| `lr_pi` | `0.05` | 自适应学习率，初始值较高 |
| `lr_v` | `1e-3` | 固定值 |
| `batch_size` | `None` | Actor 更新使用全量 batch (不分 minibatch) |
