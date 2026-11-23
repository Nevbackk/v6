## 摘要（Overview）

本 PR 在完整保留现有 `V6-NoHistoricalState`（`Gain` 与 `DirectCorrection`）全部代码、数据接口与二体 RK4 物理引擎完全向后兼容的基础上，针对**单平台仅测角（LOS-only）空间目标六维轨道在线状态估计**中的核心可观测性与鲁棒性瓶颈，新增两套完整、互补的模型方案与分层实验流水线：

1. **模型方案一（Model Plan V1）—— 物理逆向重投影信赖域修正与多假设主动消歧滤波**：
   - **单点物理预条件修正族（`LTR-LM` / `JF-BR` 及经典对照 `CL-GN1` / `CL-WindowLM`）**：利用二体轨道逆向积分将历史关键帧残差统一对齐至当前状态切空间，由轻量网络预测关键帧权重与六维各向异性阻尼，嵌入可微 $6\times 6$ Levenberg–Marquardt 正规方程求解器，并前置自标定角噪声死区保护。
   - **多假设生成式主动消歧旗舰（`FD-ASF`）**：在静态参考历元的“2 维全局距离歧义 + 4 维局部切角”物理图卡上，训练显式密度条件归一化流直接逼近**最小化下一窗口视线预测方差的分歧加权提议分布** $q_k^\star(\boldsymbol{\theta}) \propto \pi_k(\boldsymbol{\theta})\|\mathbf{h}_{k+1}(\boldsymbol{\theta})-\bar{\mathbf{h}}_{k+1}\|_2$，配合增量似然粒子更新与主峰集中度门控安全发布。
2. **模型方案二（Model Plan V2：`ASD-SC`）—— 面向含未知误差初值的渐进式子空间盲去噪与闭环自纠正滤波**：
   - 放弃“在早期短弧强迫一步猜出全部六维真值”的病态监督，将连续 6 个观测块（每块 $300\text{ s}$ 观测 + $300\text{ s}$ 空档）的序列延长过程建模为**按局部几何可观测度逐级解锁的渐进除噪链**，结合奇异子空间有界正收缩算子、潜伏弱方向跨空档薄 QR 前传与受限正交写回、自生成漂移状态回放（DAgger）以及多时距物理预测损失，实现好轨不伤、弱方向跨块解锁收敛与长程预测相位自纠正。

---

## 一、研究背景与直击痛点：现有方法为什么陷入两难？

在单平台仅测角定轨任务中，观测站每一时刻仅提供一个单位视线向量 $\mathbf{u}_k = (\mathbf{r}_k - \mathbf{s}_k)/\|\mathbf{r}_k - \mathbf{s}_k\|$。六维状态（三维位置 + 三维速度）在单历元只有两个横向角度自由度受直接约束，斜距 $\rho$ 与径向速度天然缺失；在单块短弧内，成比例同时缩放距离与横向相对速度几乎不改变视线方向与角速度。

这一物理本质导致现有两类主流方法各自撞上明显瓶颈：

1. **“均值误差”与“典型好轨精度”顾此失彼**：
   - 经典 EKF 在初轨 $B_0$ 误差较小时能够保持很高的典型精度（中位数 `P50` 小），但面对大初轨误差时易因一阶线性化失配而发散，拉高整体均方误差（`Scaled MSE`）。
   - 现有端到端神经滤波器（如 `KalmanNet`、无约束直接修正头）为了在跨越数个数量级的混合初差训练集上压低全局均方误差，倾向于输出激进修正步长。其代价是：**当输入本身已是高精度初轨（例如亚公里级初差）时，网络依然强行施加非零修正，把原本准确的好轨道推坏**，表现为全局 `Scaled MSE` 下降但中位数 `P50` 反而显著恶化。
2. **早期短弧“一步到位”监督违背几何可观测性**：
   - 在观测刚开始的几十秒内，6 个状态自由度中有 4 个尚处于几何近似零空间；对每一帧直接施加完整六维真值回归损失，等于强迫网络在信息不足时“硬猜”距离与速度。面对短弧多解（或连续距离脊线），均方误差最优解会坍缩到多条合法轨道的条件均值——几何上恰好处在非物理的轨道真空区，引发步长过冲或恒等复制捷径。
3. **单改速度或单时刻训练引发跨 $300\text{ s}$ 空档的相位漂移**：
   - 在历史重投影实验中，仅修正速度虽能保留当前帧位置不变，但在位置误差尚未消除时单独把速度“修准”，可能打破原有的位置-速度误差相位抵消，使跨过 $300\text{ s}$ 静默空档后的自由传播位置误差反而成倍放大；同时，仅在静态样本上单步训练的模型从未见过自身残余误差跨空档传播后的真实漂移分布。

---

## 二、核心思想与模型设计一（Model Plan V1）

### 2.1 单点物理预条件修正器：`LTR-LM` 与 `JF-BR`（`v6/models/v1_ltr_lm.py`）

**核心直觉**：二体引力动力学精确已知且无过程噪声。与其让黑盒循环网络从历史隐藏状态中隐式记忆轨道运动，不如直接把当前候选状态 $\tilde{\mathbf{x}}_k$ **沿二体方程逆向积分回溯**到已到达的因果关键帧集合 $\mathcal{W}_k$（上限 $W=16$）。这样所有历史观测残差都被严格锚定在同一个当前状态上：位置误差表现为整体平移，横向速度误差则随回溯时间跨度 $\Delta t_j = t_j - t_k$ 线性拉开力臂，几何物理意义一目了然。

具体设计包含三步：
1. **自标定角噪声死区门控（守住好轨底线）**：利用当前因果前缀内的同站短跨度不等距观测三元组（天然兼容 `pois10` 非均匀采样），消去线性角运动趋势以在线估计测角噪声本底 $\tau_k$。当窗口角残差中位数 $a_k \le \tau_k$ 时，门控系数 $\alpha_k = 0$，模型严格保持零步更新，从结构上杜绝把好轨道改坏。
2. **学习型信赖域一步微分改正（`LTR-LM`）**：在当前视线局部无量纲坐标（对数斜距 + 球面指数回缩切角 + 时间归一化速度，局部雅可比 $B_k$）下，固定观测减预测切残差 $\boldsymbol{\nu}_0 \in \mathbb{R}^{2W}$ 并构建经符号严格核验的前向设计矩阵 $A = -D_{\mathbf{y}}\boldsymbol{\nu}(\mathbf{0}) \in \mathbb{R}^{2W\times 6}$（满足 $\boldsymbol{\nu}(\mathbf{y}) \simeq \boldsymbol{\nu}_0 - A\mathbf{y}$）。轻量网络（约 `15k` 参数）不再直接输出状态增量，而是预测每帧可靠性权重 $\mathbf{w}_\theta \in (w_{\min}, 1]^W$ 与六维对角阻尼 $\Lambda_\theta$，通过精确可微的 $6\times 6$ Cholesky 分解求解正规方程：
   $$\left(A^\top W_b A + \Lambda_\theta\right)\mathbf{y}_k = \alpha_k A^\top W_b \boldsymbol{\nu}_0$$
   同时内置三种严格区分的状态写回模式以供实证裁决：**六维同步更新（`joint_6d`）**、**用 Schur 补消去未知位置后仅更新速度（`schur_vel`）**、以及**锁死位置直接拟合速度（`lock_pos_vel`）**。
3. **受控对照体系**：同步实现无需状态转移矩阵的无雅可比有界修正头 `JF-BR`（齐次无偏置结构，零残差必出零步）以及共享完全相同窗口与死区门控的零参数经典基准 `CL-GN1-Matched` 与多步 `CL-WindowLM`，确保任何“学习增益”均在严格公平的同信息预算下接受检验。

### 2.2 多假设主动消歧生成式滤波：`FD-ASF`（`v6/models/v1_fd_asf.py`）

**核心直觉**：当短弧观测在物理上确实存在多条竞争轨道（例如远近不同距离分支在首块内角残差几乎同样小）时，强行输出一个点必然出错。正确做法是在短弧期用极少量高价值粒子**把真正有分歧的竞争轨道保留下来**，等跨过 $300\text{ s}$ 空档新视角出现时一举消歧。

`FD-ASF` 由三个紧密咬合的支点组成：
1. **采样率不变的轨迹统计前端与 $2+4$ 物理歧义图卡**：在每个观测块内用正交 Legendre 多项式提取可加充分统计矩阵 $(H_s, B_s)$，将 `dense / stride5 / stride20 / pois10` 四种不同采样率统一映射为定长几何上下文 $\mathbf{c}_k$。在静态参考历元 $t_{\text{ref}}$，将六维轨道拆分为 **2 维全局距离歧义坐标 $\mathbf{z}$** 与 **4 维毫弧度级切角松弛坐标 $\boldsymbol{\eta}$**——跨块基线张开时启用双锚点通用变量 $M=0$ Lambert 图卡（因近地点高于地表时轨道周期 $>5069\text{ s} > 3300\text{ s}$，整圈数恒为 0），短弧或近共线时平滑回退至具有闭式雅可比行列式的单历元 Attributable $(\ln\rho, \dot{\rho})$ 图卡，并严格乘入坐标变换体积元 $|\det DF_b|$。
2. **面向未来视线消歧的最优提议学习（零闭环 BPTT）**：如果直接拟合当前后验 $\pi_k$，在实时极小粒子预算（如 $P=12$）下，粒子会挤在主峰内部下一时刻视线几乎重合的冗余位置，漏掉当前概率仅 $10\%$ 但跨空档后视线显著偏离的竞争分支。由自归一化重要性采样（SNIS）渐近方差最小化与 Cauchy–Schwarz 不等式可知，针对下一已知测站位置 $(t_{k+1}, \mathbf{s}_{k+1})$（不读取任何未来观测值 $\mathbf{u}_{k+1}$）的最优提议分布为：
   $$q_k^\star(\boldsymbol{\theta}) \propto \pi_k(\boldsymbol{\theta})\,\|\mathbf{h}_{k+1}(\boldsymbol{\theta}) - \bar{\mathbf{h}}_{k+1}\|_2$$
   我们采用 4 层显式密度条件耦合流（`ConditionalRealNVP2D`，`<35k` 参数）配合防御性混合底分布 $q_{\text{mix}} = (1-\epsilon)q_\psi + \epsilon q_{\text{phys}}$ 直接拟合该目标：单次前向即可采样并精确求取对数密度，新块到达时仅增量累乘新观测似然并对高权重模态执行 $6\times 6$ LM 局部精修。
3. **三条件主峰集中度安全发布器**：保护先验轨迹独立驻留点状态候选池；仅当“先验超出噪声死区 $\wedge$ 最优模态吸引盆地后验权重占比 $W_{\text{dom}} \ge \gamma_{\text{dom}}$ $\wedge$ 自由度惩罚似然净胜”三条件同时满足时才切换对外输出，多峰难分时稳守先验不盲跳，视差破局后一步锁定真轨。

---

## 三、核心思想与模型设计二（Model Plan V2：`ASD-SC`）

针对**初始值 $B_0$ 带有未知幅度误差（$10^{-1}\sim 10^4\text{ km}$）且自回归预测状态在跨空档传播中持续演化**的实际场景，`v6/models/v2_asd_sc.py` 与 `v6/training_v2.py` 提出了 **自适应子空间渐进盲去噪与自纠正模型（`ASD-SC`）**。

### 3.1 核心范式转换：用“物理观测时间延长”充当“渐进分步去噪时钟”

我们不再要求模型在短弧早期一步猜出六维真值，而是顺应物理可观测性的自然涌现顺序，在全序列最多 18 个宏观检查点（每块 `+30 s, +150 s, +300 s`）上**分期、分方向逐步剥离初值与预测噪声**：
- **超短弧期**：仅剥离强可观的二维横向位置误差，未可辨的径向距离与速度保持冻结；
- **块内展开期**：角速度信噪比建立，逐步收缩切向速度/距离比值误差；
- **跨 $300\text{ s}$ 空档与多块累积期**：长基线视差与二体引力曲率显现，逐级解锁径向距离与径向速度，并吸收前序步骤残留的微漂移。

### 3.2 四大核心机制

1. **双窗口分离池化与奇异子空间有界盲去噪算子**：
   - 将“历史旧窗口残差”与“上次发布后新到达段残差（尤其是跨空档后同化前的新观测）”分别独立池化，防止跨空档失配信号被旧帧中位数淹没。
   - 在敏感度矩阵的奇异子空间上，先由分位数头盲估各方向的对数误差幅度（`0.5 / 0.9` 分位数，由 Pinball 损失监督），再输出有界正收缩系数 $g_i = g_{\min} + (1 - g_{\min})\sigma(f_\theta) \in [g_{\min}, 1]$ 与限幅岭步长 $c_i$。在噪声死区内 $\chi_i=0$ 严格保先验；在已激活的可辨方向上，正下界 $g_{\min}>0$ 彻底阻断“一律输出 0 复制坏先验”的恒等捷径，且保证单步误差同向比例收缩、绝不反向过冲。
2. **潜伏“假距离”风险跨空档薄 QR 前传与受限子空间写回（零未授权泄漏）**：
   - 短弧期不可辨的弱奇异方向基 $V_p$ 暂不更新，随主状态跨空档传播时通过薄 QR 分解 $D^{-1}\Phi(t_m, t_{m-1}) D V_p = Q R$ 前传正交基 $Q$ 与尺度混合因子 $R$。
   - 新观测到达时，先消去普通可辨方向 $V_n$ 计算新增条件敏感度矩阵 $K_p = (I - P_n)A_+ V_p$，严格按实测几何条件识别新近可辨轴 $R_{\text{rel}}$，构造授权写回正交基 $E_a = [V_n, V_p R_{\text{rel}}]$（仍不可辨补空间为 $E_h$）。微步内仅对受限矩阵 $A E_a$ 做奇异值分解并通过 $\Delta\mathbf{y} = E_a \Delta\mathbf{y}_{\text{red}}$ 写回六维，从代数上严格保证 **$E_h^\top \Delta\mathbf{y} \equiv \mathbf{0}$**（即便全矩阵奇异轴发生旋转，也不会向未解锁方向泄漏任何伪更新）。对于大幅距离失配样本，进一步提供子弧 A（拟合重锚候选）$\to$ 子弧 B（盲测因果择优）$\to$ 子弧 C（样本外无偏评价）的两级恢复状态机。
3. **自生成漂移状态聚合回放（DAgger）+ 跨空档双检查点连通展开**：
   - 周期性用当前模型从真实 $B_0$（及同观测配对的“好先验 / 显性坏先验 / 等视线角速度假距离先验”反事实干预起点）连跑 6 个观测块，收集模型真实访问的含漂移状态进入回放池；抽取相邻双检查点对（显式覆盖跨 $300\text{ s}$ 空档对），在首点入口截断梯度（`stop_gradient`），两点间保持真实二体传播与探针重算连通反传，直接教会模型修补自己上一步留下的预测漂移。
4. **多时距物理预测风险 + 可开关微步增量渐进监督 + 好先验单边保护**：
   - 物理主损失同时惩罚当前六维误差与纯二体自由传播 `+60 s`、`+300 s` 后的三维位置误差，迫使网络协调位置与速度修正的轨道相位；
   - 提供可独立开关消融的微步增量渐进辅助项 $\mathcal{L}_{\text{prog},\ell}$，按当前实测奇异谱收缩阵 $T_\ell = V_{\text{phys}}\operatorname{diag}(s_i^2/(s_i^2+\gamma^2))V_{\text{phys}}^\top$ 监督**单步增量** $(\mathbf{y}_{\ell+1}-\mathbf{y}_\ell) \approx \operatorname{sg}(T_\ell\mathbf{d}_\ell)$；
   - 对好先验施加带 `stop_gradient` 保持基准的单边防擦除惩罚 $[\mathscr{E}_m(\mathbf{x}_m^+) - \operatorname{sg}(\mathscr{E}_m(\mathbf{x}_m^-)) - \delta_G]_+$。

---

## 四、新增代码结构与严格评测规范映射

| 文件路径 | 功能说明 |
|---|---|
| `v6/models/reprojection.py` | 离散 RK4 解析切线性状态转移矩阵 $\Phi(t_j,t_k)$、全球面无奇异正交切标架、无抵消增量球面回缩、符号核验重投影算子与兼容 `pois10` 的不等距三元组角噪声标定 |
| `v6/data/splits.py` | 按 `source_id` 严格隔离的 `train_fit / train_calib / dev_select` 切分、跨 $300\text{ s}$ 空档分块器、子弧 A/B/C 划分及含 `Block 2 + 30 s` 公共主终点的检查点调度器 |
| `v6/eval_metrics.py` | 四节奏独立分表（`dense / stride5 / stride20 / pois10`）、`source_id` 等权汇总、五档初差、相对同历元未更新先验的好轨伤害率、防挑选偏差方向余弦三元组、按 `source` 聚类成对 Bootstrap 95% CI 与闭环割线放大率审计 |
| `v6/models/v1_ltr_lm.py` | Model V1 单点族：`CL-GN1-Pure / Matched`、`CL-WindowLM`、`CL-FullPrefixWLS`、`JF-BR-Vel / 6D`、`LTR-LM`（`joint_6d / schur_vel / lock_pos_vel`）及在线闭环封装 |
| `v6/models/v1_fd_asf.py` | Model V1 旗舰 `FD-ASF`：可加 Legendre 轨迹编码器、$2+4$ 双锚点 Lambert + 单历元 Attributable 图卡、4 层条件 RealNVP 预测分歧提议流、防御性混合与集中度门控发布器 |
| `v6/models/v2_asd_sc.py` | Model V2 `ASD-SC`：奇异子空间有界盲去噪、0.5/0.9 分位数头、跨空档薄 QR 前传、$K_p$ 正交消元解锁与 $E_a$ 受限写回、A/B/C 重锚头及五组受控消融模式开关 |
| `v6/training_v1.py` / `v6/training_v2.py` | V1 与 V2 完整训练、DAgger 回放、跨空档双步连通展开与统一分层验证入口 |
| `scripts/train_v1.py`, `validate_v1.py`, `train_v2.py`, `validate_v2.py` | 与现有 `train_gain.py` / `train_direct.py` 参数习惯一致的单模型 CLI 入口 |
| `scripts/run_experiment_v1.{sh,ps1}`, `run_experiment_v2.{sh,ps1}` | 一键执行 V1 与 V2 全阶段基准、多随机种子训练、受控消融矩阵及锁定 `VAL` 成对复验的跨平台脚本 |
| `tests/test_v1_v2_models.py` | 12 项核心数学性质与端到端训练/回放单元测试（连同原有 `tests/test_smoke.py` 共 13 项测试全部通过） |

---

## 五、本地验证结果与复现运行命令

### 5.1 已通过的本地数学与端到端单元测试（FP64）
运行 `python -s -m unittest discover -s tests -v`（13/13 全部 `OK`），已严格核验：
- 原有 V6 `Gain` 与 `DirectCorrection` 训练、状态加载与一致性测试无任何回归；
- 离散 RK4 解析切线性状态转移矩阵 $\Phi$ 与自动微分精确雅可比相对误差 `< 1e-13`；
- 正规方程下降符号检验（标量退化例残差平方 `1 -> 0.25`）及三维二体有限差分 `-D_y nu(0)` 相对误差 `< 1e-6`；
- `LTR-LM` 中 `schur_vel` 与 `joint_6d` 速度块机器精度等价（`< 1e-12`），零残差好轨下 `CL-GN1-Matched / JF-BR / LTR-LM / ASD-SC` 精确零步恒等保持；
- `FD-ASF` 连续时间轨迹统计量精确分批可加性（`< 1e-13`）、`Attributable` 图卡解析体积元对标自动微分（`< 1e-7`）、$M=0$ Lambert 闭环命中精度及 90/10 双分支预测分歧权重主动抬升；
- `ASD-SC` 跨空档薄 QR 正交性、受限写回未授权方向零泄漏（$\|E_h^\top\Delta\mathbf{y}\| < 10^{-14}$）、渐进微步增量标签（`0.5 -> 0.25 -> 累计 0.75`）、短弧假距离反事实初轨视线/角速度不变性及跨空档双检查点连通反传。
*(注：全量 G4 43,783 训练轨迹与 10,857 验证轨迹上的性能对比待挂载冻结数据集后按预注册脚本实测产出。)*

### 5.2 快速运行与验证命令

```bash
# 1. 运行全量数学与端到端单元测试（无需外部数据集）
python -s -m unittest discover -s tests -v

# 2. 单独训练并验证 Model V1 主模型（LTR-LM / FD-ASF）
python scripts/train_v1.py    --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v1_ltr_lm --arch ltr_lm --writeback-mode joint_6d --loss-type log_charbonnier --device cuda
python scripts/validate_v1.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v1_ltr_lm --arch ltr_lm --resume runs/v1_ltr_lm/best.pt --eval-split paired_val --device cuda

python scripts/train_v1.py    --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v1_fd_asf --arch fd_asf --num-particles 12 --device cuda
python scripts/validate_v1.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v1_fd_asf --arch fd_asf --resume runs/v1_fd_asf/best.pt --eval-split paired_val --device cuda

# 3. 单独训练并验证 Model V2 完整主模型（ASD-SC）
python scripts/train_v2.py    --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v2_asd_sc --ablation-mode full --horizons-sec "60,300" --lambda-prog 0.25 --device cuda
python scripts/validate_v2.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v2_asd_sc --ablation-mode full --resume runs/v2_asd_sc/best.pt --eval-split paired_val --device cuda

# 4. 或一键执行 V1 / V2 完整分阶段实验与消融流水线
PREPARED_ROOT=/path/to/prepared PROTOCOL_DIR=/path/to/protocol bash scripts/run_experiment_v1.sh
PREPARED_ROOT=/path/to/prepared PROTOCOL_DIR=/path/to/protocol bash scripts/run_experiment_v2.sh
```
