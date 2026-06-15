# Training Guide: Q-Transfer MLP Routing

## Prerequisites

```bash
# System packages
sudo apt-get install libzmq3-dev python3-zmq

# Python packages
pip3 install torch numpy pyzmq
```

## 1. Compile gem5 (ZMQ Mode)

```bash
cd /path/to/gem5

# Ensure ZMQ mode (both defines commented out)
# File: src/mem/ruby/network/garnet/zmq_functions.cc
#   // #define ADQN_NATIVE_INFERENCE
#   // #define ADQN_Q_TRANSFER

# Clean build
rm -rf build/Garnet_standalone
scons build/Garnet_standalone/gem5.opt -j4

# Verify ZMQ is linked
ldd build/Garnet_standalone/gem5.opt | grep zmq
# Should show: libzmq.so.5 => /lib/x86_64-linux-gnu/libzmq.so.5
```

## 2. Train Base Model (Uniform Random Traffic)

### Start Python Server
```bash
fuser -k 7777/tcp 2>/dev/null; sleep 1
scons build/Garnet_standalone/gem5.opt -j4 2>&1 | tail -2

nohup python3 -u configs/network/adqn/madrl_server_l02.py \
    --mesh-cols=8 --mesh-rows=8 --inj-rate=0.20 --save-interval=20000 \
    --save=checkpoints/mlp_l004_uniform.pt \
    > logs/uniform_train.log 2>&1 &
sleep 6
```

### Run gem5 Training Loop (Multiple Injection Rates)
```bash
# Progressive training: low rate → high rate
RATES=(0.05 0.10 0.15 0.20 0.25 0.30)
for rate in "${RATES[@]}"; do
    for i in $(seq 1 10); do
        build/Garnet_standalone/gem5.opt \
            configs/example/garnet_synth_traffic.py \
            --num-cpus=64 --num-dirs=64 --mesh-rows=8 \
            --network=garnet --topology=Mesh_XY \
            --sim-cycles=2000000 --synthetic=uniform_random \
            --injectionrate=$rate --routing-algorithm=3 2>&1 | grep "Exiting"
        echo "rate=$rate round=$i step=$(tail -1 logs/uniform_train.log | grep -oP 's=\K[0-9]+')"
    done
done
```

## 3. Fine-Tune on Target Traffic Pattern

### Transpose Fine-Tuning Example
```bash
# Start from uniform checkpoint, fine-tune on transpose
fuser -k 7777/tcp 2>/dev/null; sleep 1

nohup python3 -u configs/network/adqn/madrl_server_l02.py \
    --mesh-cols=8 --mesh-rows=8 --inj-rate=0.30 --save-interval=20000 \
    --checkpoint=checkpoints/mlp_l004_uniform.pt \
    --epsilon=0.05 \
    --save=checkpoints/mlp_l004_uniform_transpose.pt \
    > logs/transpose_ft.log 2>&1 &
sleep 6

# Run 100 rounds at fixed rate (or chain through multiple rates)
for i in $(seq 1 100); do
    build/Garnet_standalone/gem5.opt \
        configs/example/garnet_synth_traffic.py \
        --num-cpus=64 --num-dirs=64 --mesh-rows=8 \
        --network=garnet --topology=Mesh_XY \
        --sim-cycles=2000000 --synthetic=transpose \
        --injectionrate=0.30 --routing-algorithm=3 2>&1 | grep "Exiting"
    echo "round=$i step=$(tail -1 logs/transpose_ft.log | grep -oP 's=\K[0-9]+')"
done
```

## 4. Export Weights for Native C++ Inference

```bash
# Export MLP weights and warm Q-values
python3 << 'PYEOF'
import torch, sys
sys.path.insert(0, 'configs/network/adqn')
from madrl_server_l02 import MLPQNet

net = MLPQNet()
net.load_state_dict(torch.load('checkpoints/mlp_l004_transpose_ft.pt', map_location='cpu'))
net.eval()

# Export to C header files
# (see madrl_server_l02.py export_weights and export_warm_q functions)
PYEOF
```

## 5. Test with Native Inference

```bash
# Switch to native mode
# File: src/mem/ruby/network/garnet/zmq_functions.cc
#   #define ADQN_NATIVE_INFERENCE
#   #define ADQN_Q_TRANSFER

# Rebuild
rm -f build/Garnet_standalone/gem5.opt
scons build/Garnet_standalone/gem5.opt -j4

# Run test (no Python server needed)
build/Garnet_standalone/gem5.opt \
    configs/example/garnet_synth_traffic.py \
    --num-cpus=64 --num-dirs=64 --mesh-rows=8 \
    --network=garnet --topology=Mesh_XY \
    --sim-cycles=2000000 --synthetic=transpose \
    --injectionrate=0.30 --routing-algorithm=3

# Extract latency from stats
grep "average_flit_latency" m5out/stats.txt
```

## 6. Test with XY Baseline

```bash
# XY routing (routing-algorithm=1)
for rate in 0.03 0.06 0.09 0.12 0.15 0.18 0.21 0.24 0.27 0.30; do
    build/Garnet_standalone/gem5.opt \
        configs/example/garnet_synth_traffic.py \
        --num-cpus=64 --num-dirs=64 --mesh-rows=8 \
        --network=garnet --topology=Mesh_XY \
        --sim-cycles=2000000 --synthetic=transpose \
        --injectionrate=$rate --routing-algorithm=1 2>&1 | tail -1
    grep "average_flit_latency" m5out/stats.txt
done
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_TOKENS` | 3 | Current + 2 toward-neighbor tokens |
| `TOKEN_DIM` | 18 | Position(4) + hop(4) + credit(4) + boundary(1) + inport(1) + self_cong(4) |
| `LAMBDA` | 0.04 | Q-Transfer coefficient |
| `BATCH_SIZE` | 128 | Replay buffer batch size |
| `REPLAY_CAPACITY` | 200,000 | Experience replay buffer size |
| `GAMMA` | 0.99 | Discount factor |
| `LR` | 1e-3 | Learning rate |
| `EPS_START→END` | 1.0→0.05 | ε-greedy exploration |
| `EPS_DECAY` | 0.9995 | ε decay rate |
| `TARGET_UPDATE` | 500 | Target network update interval |
| `SIM_CYCLES` | 2,000,000 | gem5 simulation cycles per round |

## Important Notes

1. **Always verify binary linkage**: `ldd gem5.opt | grep zmq` must show libzmq
2. **Keep server running across rounds**: Server accumulates training; do NOT restart between rounds
3. **ε=0.05 for fine-tuning**: Pure exploitation from a pre-trained model
4. **Multi-rate chaining**: Start from low injection rates and progress to higher ones
5. **warm_q.h regeneration**: Required after every training session for native inference
6. **Binary rebuild**: After `scons` says "up to date" when you changed zmq_functions.cc, use `rm -f gem5.opt` first
7. **ZMQ modes**: Server must match binary mode (ZMQ server ↔ ZMQ binary, Native ↔ no server)

## Common Issues

### gem5 segfault with ADQN routing
- Check `ldd gem5.opt | grep zmq` — libzmq must be linked
- If not linked: `rm -rf build/Garnet_standalone && scons ...`

### Server shows 0 steps after training
- Check `--routing-algorithm=3` (not 1 for XY)
- Check zmq_functions.cc defines (both should be commented for ZMQ mode)

### Training produces few steps on congested traffic
- Increase ε to 0.3 for exploration
- Use lower injection rate first
- Try more rounds (100+) at a single rate
