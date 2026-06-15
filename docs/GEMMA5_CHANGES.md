# gem5 Source Code Modifications

All changes are relative to gem5 v24.0.0.1 stable branch.

## Modified Files

### `src/mem/ruby/network/garnet/CommonTypes.hh`
- Added `ADQN_ = 3` to `RoutingAlgorithm` enum

### `src/mem/ruby/network/garnet/GarnetNetwork.hh`
- Added ADQN ZMQ port members (`m_adqn_zmq_decision_port`, `m_adqn_zmq_experience_port`)
- Added injection rate and detour multiplier members
- Added global latency EMA member (`m_global_latency_ema`)
- Added per-router Q-buffer (`m_router_prev_q[256][4]`) for Q-Transfer
- Added getter/setter methods: `getRouterPrevQ()`, `setRouterPrevQ()`
- Added `getGlobalLatencyEMA()`, `updateGlobalLatencyEMA()`

### `src/mem/ruby/network/garnet/GarnetNetwork.cc`
- Constructor: Initialize ADQN parameters from SimObject params
- Initialize Q-buffer from `warm_q.h` (warm-start values)
- Implement `getRouterPrevQ()`, `setRouterPrevQ()`, `updateGlobalLatencyEMA()`

### `src/mem/ruby/network/garnet/GarnetNetwork.py`
- Added SimObject parameters: `adqn_zmq_decision_port`, `adqn_zmq_experience_port`, `adqn_injection_rate`, `adqn_detour_multiplier`

### `src/mem/ruby/network/garnet/zmq_functions.cc` (NEW)
- Dual-mode inference: ZMQ (training) + Native C++ (deployment)
- ZMQ mode: Send tokens+mask to Python server, receive action
- Native mode: BN→FC(54→32)→ReLU→FC(32→16)→ReLU→FC(16→4)
- Q-Transfer in native mode: Direction-matched neighbor Q-value fusion

### `src/mem/ruby/network/garnet/zmq_functions.hh` (NEW)
- Function declarations for ZMQ/native inference interfaces
- Batch inference support structures
- Stub implementations when HAVE_ZMQ is not available

### `src/mem/ruby/network/garnet/adqn_routing.cc` (NEW)
- `buildTokenStates()`: 3-token state representation
- `buildActionMask()`: Smart detour mask (credit < 0.25 → all directions)
- `computeReward()`: Credit-scaled progress reward
- `actionToOutport()`: Action index → physical outport mapping
- `getTowardNeighborIds()`: Toward-destination neighbor router IDs

### `src/mem/ruby/network/garnet/adqn_routing.hh` (NEW)
- Constants: `ADQN_NUM_TOKENS=3`, `ADQN_TOKEN_DIM=18`, `ADQN_NUM_ACTIONS=4`
- Token dimension indices for all 18 features
- Function declarations

### `src/mem/ruby/network/garnet/RoutingUnit.cc`
- Added `case ADQN_:` in `outportCompute()`
- Added `outportComputeADQN()` function: Builds tokens → ZMQ inference → reward → experience

### `src/mem/ruby/network/garnet/RoutingUnit.hh`
- Added `outportComputeADQN()` declaration

### `src/mem/ruby/network/garnet/NetworkInterface.cc`
- Added `m_net_ptr->updateGlobalLatencyEMA()` call in `incrementStats()`

### `src/mem/ruby/network/garnet/SConscript`
- Added `adqn_routing.cc` and `zmq_functions.cc` to build sources

### `SConstruct`
- Added ZMQ detection (libzmq check with HAVE_ZMQ define)

### `configs/network/Network.py`
- Added `routing-algorithm=3` option for ADQN
- Added `--adqn-zmq-decision-port` and `--adqn-zmq-experience-port` CLI arguments
- Added `--adqn-detour-multiplier` CLI argument

### `configs/example/garnet_synth_traffic.py`
- Added ADQN parameter wiring: `adqn_injection_rate`, `adqn_detour_multiplier`
