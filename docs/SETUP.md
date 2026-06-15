# ADQN 路由系统 —— 从零配置指南

## 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  gem5 仿真器 (C++)                                                    │
│                                                                       │
│  RoutingUnit::outportComputeADQN()                                    │
│    │                                                                  │
│    ├── 4 方向 credit (E/W/N/S) + 14D 自身特征                          │
│    ├── 收集当前 + 4 邻居的 state (5 tokens × 14D)                      │
│    ├── 生成 action_mask（边界无效=0，有效=1）                           │
│    ├── zmq_functions.h/cc    ← ZMQ 封装层                             │
│    ├── adqn_routing.h   ← build_token_states() + 决策 + 奖励           │
│    │                                                                  │
│    │  ZMQ REQ :7777 ─── 5-token + mask ──►  Python server            │
│    │          ◄── action (0~3) ───────────                             │
│    │  ZMQ PUSH :7778 ─── transition (+reward) ─►  Python server       │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  Python DQN Server                                                    │
│                                                                       │
│  server.py                                                            │
│    ├── Decision Thread   (REP :7777)                                  │
│    │     parse_state() → 5×14 tensor + 4D mask                        │
│    │     DuelingQNet: Embed → +Pos → Attn×2 → Token[0] → Q×4 (masked) │
│    ├── Experience Thread (PULL :7778)  收 transition + 训练            │
│    └── ReplayBuffer + Double DQN + Action Masking                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 状态向量（每个 token 14D，纯自身特征，全部可从 gem5 获取）

| 维度 | 名称 | gem5 获取方式 | 含义 | 范围 |
|------|------|-------------|------|------|
| [0] | rx_norm | `id % cols / (cols-1)` | 当前 X | [0,1] |
| [1] | ry_norm | `id / cols / (rows-1)` | 当前 Y | [0,1] |
| [2] | dx_norm | `dest_id % cols` | 目的 X | [0,1] |
| [3] | dy_norm | `dest_id / cols` | 目的 Y | [0,1] |
| [4] | hops_taken_norm | 需埋点 / max_hops | 已跳数 | [0,1] |
| [5] | hops_remain_norm | `|dx-rx|+|dy-ry| / max` | 剩余距离 | [0,1] |
| [6] | hops_remain_x | `abs(dest_x - x)` raw | X 剩余步数 | [0,N] |
| [7] | hops_remain_y | `abs(dest_y - y)` raw | Y 剩余步数 | [0,N] |
| [8] | credit_E | `OutputUnit::get_credit_count(vc)` 求和归一化 | East VC 空闲 | [0,1] |
| [9] | credit_W | 同上 | West VC 空闲 | [0,1] |
| [10] | credit_N | 同上 | North VC 空闲 | [0,1] |
| [11] | credit_S | 同上 | South VC 空闲 | [0,1] |
| [12] | is_boundary | `x==0 \|\| x==cols-1 \|\| y==0 \|\| y==rows-1` | 是否边界 | {0,1} |
| [13] | inport_dir | `inport_dirn` 编码 0=E 0.25=W 0.5=N 0.75=S 1=L | 数据包来向 | [0,1] |

5 个 token = 当前 + E/W/N/S 邻居（同理 14D，从邻居视角）。
邻居 inport = 连接反方向（E 邻居←W, W 邻居←E, N 邻居←S, S 邻居←N）。
**邻居信息完全通过 self-attention 融入当前 token。**

### 动作空间（4 方向，Local 由硬件直接处理）

| Action | 方向 | 含义 |
|--------|------|------|
| 0 | East | 东（x+1） |
| 1 | West | 西（x-1） |
| 2 | North | 北（y+1） |
| 3 | South | 南（y-1） |

到达目标路由器时，gem5 在调 DQN 之前直接走 Local 端口。

### Action Mask 机制

mask **仅屏蔽物理上不存在边界**的方向：

| 方向 | 屏蔽条件 |
|------|---------|
| East | `my_x == num_cols-1`（最右列） |
| West | `my_x == 0`（最左列） |
| North | `my_y == num_rows-1`（最上行） |
| South | `my_y == 0`（最下行） |

### 奖励函数

| 情况 | Reward |
|------|--------|
| 选择边界外方向 | **-1.0** |
| 最短路径方向 | **+0.3** |
| 非最短路径方向 | **-0.4**（绕路惩罚） |

### 网络结构

```
Token 0 (当前, 14D 自身) ─┐
Token 1 (East,  14D 自身) ─┤                              ┌──────────────┐
Token 2 (West,  14D 自身) ─┼─→ Embed(14→32)+Pos ─→ Attn×2 ─→ Token[0] ─→│ Value Head   │→ V
Token 3 (North, 14D 自身) ─┤    ↑ 邻居信息在此融入         │              │              ──→ Q×4
Token 4 (South, 14D 自身) ─┘    ↑                         └→│ Adv Head    │→ A×4   (mask -inf)
                                                             └──────────────┘
```


