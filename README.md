# Q-Transfer Enhanced MLP Routing for Network-on-Chip

**Q-Transfer**: Direction-matched, previous-step Q-value sharing between neighboring routers for congestion-aware adaptive routing in 8×8 mesh NoC.

## Overview

Traditional MADRL routing makes each router decide independently. Q-Transfer allows routers to share compressed path-quality information (4D Q-vectors, 128 bits) with downstream neighbors, enabling **implicit multi-hop congestion awareness** without extra communication overhead.

### Key Results (vs XY Routing @ 0.30 injection rate)

| Traffic | XY Latency | MLP λ=0.04 | Improvement |
|---------|:----------:|:----------:|:-----------:|
| tornado | 92.4 cyc | **12.2 cyc** | **-87%** |
| shuffle | 113.5 cyc | **41.7 cyc** | **-63%** |
| bit_complement | 168.5 cyc | 157.0 cyc | -7% |

## Repository Structure

```
├── README.md                   # This file
├── docs/
│   ├── QTransfer_MLP_Model.md  # Complete model documentation
│   ├── TRAINING_GUIDE.md       # Step-by-step training procedure
│   ├── SETUP.md                # Environment setup
│   └── results/                # Experiment result CSVs
├── src/
│   ├── garnet/                 # Modified gem5 C++ source files
│   │   ├── zmq_functions.cc/hh # ZMQ + Native MLP inference (dual-mode)
│   │   ├── adqn_routing.cc/hh  # Token state builder, reward, action mask
│   │   ├── GarnetNetwork.cc/hh # Q-buffer, global latency EMA
│   │   ├── RoutingUnit.cc/hh   # ADQN routing entry point
│   │   ├── NetworkInterface.cc # Latency EMA update
│   │   ├── CommonTypes.hh      # ADQN_ routing algorithm enum
│   │   ├── GarnetNetwork.py    # SimObject parameters
│   │   └── SConscript          # Build configuration
│   └── server/                 # Python training servers
│       ├── madrl_server_l02.py # MLP + Q-transfer (λ=0.04) trainer
│       └── madrl_server.py     # MLP Smart (λ=0, no Q-transfer)
├── configs/                    # gem5 Python configuration files
│   ├── Network.py              # CLI routing-algorithm options
│   ├── garnet_synth_traffic.py # Traffic generator with ADQN wiring
│   └── MOESI_AMD_Base.py       # Ruby system configuration
├── scripts/
│   └── test_xy.sh              # XY baseline testing script
└── checkpoints/                # Trained model checkpoints
```

## Model Architecture

```
Input: 3 tokens × 18D = 54D (current + 2 toward-neighbor states)
  │
  ▼
BatchNorm1d(18) per-token → Flatten 54D
  │
  ▼
FC(54→32) + ReLU → FC(32→16) + ReLU → FC(16→4)
  │
  ▼
Q_local[East, West, North, South]
  │
  + λ × q_down[direction]  ← Q-Transfer (direction-matched)
  │
  ▼
Q_aug → Action Mask → Argmax → Action
```

- **Parameters**: ~2.4K float32 (~9.6 KB)
- **Q-Transfer**: λ=0.04, direction-matched previous-step neighbor Q-values
- **Smart Detour**: Credit < 0.25 → all 4 directions open
- **Credit-Scaled Reward**: credit_factor × base_progress - local_penalty - global_penalty

## Core Innovations

### 1. Direction-Matched Q-Transfer
Each action direction only fuses that direction's downstream router evaluation:
```
q_aug[East]  = q_local[East]  + λ × neighbor_prev_q[East]
q_aug[North] = q_local[North] + λ × neighbor_prev_q[North]
```
Information propagates along routing paths with λ^k decay, forming implicit multi-hop congestion awareness.

### 2. Previous-Step Q-Cache
Routers cache their own last-step q_local. Neighbors read this cached value, naturally breaking cyclic dependencies without synchronization.

### 3. Credit-Scaled Reward Function
```
progress = (old_dist - new_dist) / old_dist
credit_factor = 0.2 + 0.8 × chosen_credit
base_reward = (0.05 + 0.15 × progress) × credit_factor
total = base_reward - 0.15 × local_penalty - 0.15 × global_penalty
```

### 4. Dual-Mode Inference (ZMQ + Native C++)
- **ZMQ Mode**: Python server handles training (CTDE paradigm)
- **Native Mode**: Pure C++ inference for deployment (~1 μs/decision)

## Training Paradigm: CTDE

- **Centralized Training**: Python ZMQ server with Double DQN + Experience Replay
- **Decentralized Execution**: 64 routers share one Q-network (parameter sharing)
- **Progressive Training**: Multiple injection rates, chained checkpoints
- **Transfer Learning**: Pre-train on uniform_random, fine-tune on target patterns

## Getting Started

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for complete setup and training instructions.

## Citation

If you use this work, please cite:
```
@software{Q-Trans-NoC,
  author = {Nayomi},
  title = {Q-Transfer Enhanced MLP Routing for Network-on-Chip},
  year = {2026},
  url = {https://github.com/Nayomi-112/Q-trans-noc}
}
```
