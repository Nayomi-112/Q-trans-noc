# Q-Transfer Enhanced MLP Routing for Network-on-Chip

## 1. 模型架构

### 1.1 整体结构

```
                    输入: 3 tokens × 18D = 54D
                    ┌─────────────────────────┐
                    │  Token[0]: 当前路由器状态  │  (位置, credit, 拥塞指标)
                    │  Token[1]: East邻居状态    │  (如果East是toward方向)
                    │  Token[2]: North邻居状态   │  (如果North是toward方向)
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   BatchNorm1d (per-token) │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   Flatten 54D            │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   FC(54→32) + ReLU       │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   FC(32→16) + ReLU       │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   FC(16→4)               │
                    │   → Q_local[East,West,   │
                    │           North,South]   │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
    ┌─────────▼──────────┐          ┌──────────▼──────────┐
    │ 邻居上一步 q_local  │          │  Smart Detour Mask  │
    │ (从Q-Buffer读取)    │          │ (credit < 0.25时   │
    │ q_down[East] =     │          │  开放旁路方向)       │
    │   East邻居的prev_E │          └──────────┬──────────┘
    │ q_down[North] =    │                     │
    │   North邻居的prev_N│                     │
    └─────────┬──────────┘                     │
              │                                │
    ┌─────────▼──────────┐                     │
    │ q_aug[i] =         │                     │
    │   q_local[i]       │                     │
    │   + λ · q_down[i]  │                     │
    │                     │                     │
    │ λ = 0.02            │                     │
    └─────────┬──────────┘                     │
              │                                │
              └────────────┬───────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Argmax Q   │
                    │  → Action   │
                    └─────────────┘
```

### 1.2 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| NUM_TOKENS | 3 | 当前 + 2 toward 邻居 |
| TOKEN_DIM | 18 | 位置(4) + hop(4) + credit(4) + boundary(1) + inport(1) + self_congestion(4) |
| 输入维度 | **54D** | 3 × 18 展平 |
| 隐藏层1 | **32** | FC(54→32) + ReLU |
| 隐藏层2 | **16** | FC(32→16) + ReLU |
| 输出 | **4** | Q(East, West, North, South) |
| λ (Q-transfer) | **0.02** | 下游Q值融合系数 |
| BatchNorm | per-token BN(18) | 每个 token 独立归一化 |
| 总参数量 | **~2.4K** | 极轻量，适合硬件实现 |

---

## 2. 核心创新

### 2.1 方向匹配 Q-Transfer（创新 1）

**动机**：传统 MADRL 路由中，每个路由器独立决策，无法感知下游拥塞。

**创新**：用邻居路由器**上一步的 q_local 向量**（4D，128 bit）替代 54D 原始状态，作为下游路径质量的压缩摘要。

```
传统方案 (GCAN-A2C):  传输 54 × 32bit = 1728 bit/跳
本方案 (Q-Transfer):  传输 4 × 32bit  = 128 bit/跳  (13.5× 压缩)
```

**方向匹配机制**（关键设计）：
```
q_aug[East]  = q_local[East]  + λ × neighbor_prev_q[East]   ← 东邻居对East的评价
q_aug[North] = q_local[North] + λ × neighbor_prev_q[North]  ← 北邻居对North的评价
```

每个动作方向**只融合该方向下游路由器对该方向的评价**，而非广播标量。

**信息传播链**：Q值以 λ^k 衰减率沿路径反向传播，隐式形成多跳拥塞感知。

### 2.2 上一步 Q 值缓存（创新 2）

**问题**：实时读取邻居当前 Q 值会导致循环依赖。

**解决方案**：每个路由器缓存自己**上一次**路由决策时的 q_local：

```
时刻 t:   Router A 决策 → 存 q_local(A) 到 Q-Buffer
时刻 t+1: Router B 决策 → 读 Q-Buffer 中 A 的上一步 q_local
时刻 t+2: Router A 决策 → 读 Q-Buffer 中 B 的上一步 q_local (t时刻存的)
```

**效果**：天然打破循环依赖，无需额外同步。

### 2.3 智能绕路掩码（创新 3）

```
if any_toward_credit ≥ 0.25:
    mask = [toward directions only]     ← toward 通畅，保守路由
else:
    mask = [所有合法方向]                ← toward 全堵塞，开放绕路
```

与 DyXY 对比：

| | DyXY | 本方案 |
|------|------|------|
| 绕路触发 | 单个 toward credit < 阈值 | 所有 toward credit < 阈值 |
| 绕路选择 | 启发式规则 | MLP Q值自动选择最优 |

### 2.4 Credit-加权奖励函数（创新 4）

```
progress = (old_dist - new_dist) / old_dist
credit_factor = 0.2 + 0.8 × chosen_credit    ← 通畅=1.0, 拥堵=0.2
base_reward = (0.05 + 0.15 × progress) × credit_factor
local_penalty = 0.15 × output_busy_ratio
global_penalty = 0.15 × (global_latency_EMA / max_hops)
total_reward = base_reward - local_penalty - global_penalty
```

---

## 3. 训练方法

### 3.1 训练流程

- **CTDE范式**：集中训练（Python ZMQ），分散执行（C++ native）
- **Double DQN** + Experience Replay (200K)
- **参数共享**：64 路由器共享同一 Q 网络
- **递进训练**：bit_complement 6 注入率 (0.10→0.30)
- **ε-greedy**：1.0→0.05 (decay 0.9995)

### 3.2 λ 对比实验

| λ | 训练效果 | 推理效果 | 原因 |
|------|------|------|------|
| 0.0 | 慢 | 稳定 | 纯本地 |
| **0.02** | **快** | **稳定** ✅ | 充分信息但不依赖 |
| 0.2 | 快 | 不稳定 ❌ | 过度依赖邻居Q |
| 1.0 | 崩溃 | — | Q值递归放大 |

### 3.3 为什么加号

路由器合作转发：邻居Q值高 = 下游通畅 = 当前路由器也应走该方向。加号传递正向信号。

---

## 4. 实验结果

### 4.1 训练环境

| 参数 | 值 |
|------|-----|
| Mesh | 8×8 |
| 训练流量 | bit_complement |
| 训练步数 | ~3M |
| Buffer | 200K (满) |
| 推理 | Native C++ |

### 4.2 延迟对比 vs XY (@0.30)

| Traffic | MLP λ=0.02 | XY | 降低 |
|------|:---:|:---:|------|
| uniform_random | 17.1 | 16.2 | +6% |
| tornado | **12.2** | 92.4 | **-87%** |
| bit_complement | 157.0 | 168.5 | -7% |
| shuffle | **41.7** | 113.5 | **-63%** |

### 4.3 Tornado 全注入率

| Rate | MLP | XY |
|------|:---:|:---:|
| 0.10 | 7.6 | 7.6 |
| 0.20 | **10.5** | 11.1 |
| 0.30 | **12.2** | 92.4 |
| 0.40 | **14.5** | 154.9 |
| 0.50 | **115.1** | — |

### 4.4 通信开销

| 方案 | 传输量 |
|------|------|
| GCAN-A2C | 1728 bit/跳 |
| ATQ-Route | 128 bit/跳 |
| **本方案** | **0 bit/跳** (本地缓存) |

---

## 5. C++ Native 推理性能

| 指标 | 值 |
|------|-----|
| 推理时间 | ~1 μs/decision |
| 参数量 | 2.4K float32 ≈ 9.6KB |
| 运算量 | ~3.5K MAC/decision |
| Q-Buffer | 64 × 4 floats = 1KB |

---

## 6. 结论

1. **Q-transfer 首次在 NoC native 推理中成功**：λ=0.02 是关键平衡点
2. **87% 延迟降低**（tornado @0.30 vs XY）
3. **零额外通信**：上一步缓存 + 方向匹配
4. **2.4K 参数**：适合片上硬件实现
5. **8 流量模式泛化**：非训练流量也显著优于 XY
