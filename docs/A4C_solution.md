# ADQN 路由性能优化方案

## 问题诊断

基于 2000-cycle 公平对比实验结果：

| 问题 | 表现 | 根因 |
|------|------|------|
| 延迟随注入率线性增长 | 0.02→0.22: XY +30%, ADQN +88% | ZMQ同步往返 + toward-restriction |
| 0.24吞吐量崩溃 | 注入骤降41%, 投递率94.6% | 网络饱和, backpressure阻塞注入 |
| 0.30假低延迟 | 13.42cyc但注入仅XY的37% | 仅短路径包存活, 其余被丢弃 |

**根因链条**：
```
高注入率 → 每周期flit多 → ZMQ请求排队 → Python决策延迟增加
    → gem5等待 → flit堆积 → backpressure → 注入被阻塞
    → toward-restriction无法绕路 → 只能等 → 恶性循环
```

---

## 方案一：批量推理（推荐优先实施）

### 思路
当前每个flit每跳独立发送ZMQ请求。改为gem5侧累积N个请求，一次发送batch，Python一次forward处理整个batch。

### 架构

```
当前:                          批量:
flit1 → ZMQ → forward → a1    flit1─┐
flit2 → ZMQ → forward → a2    flit2─┼→ batch[4] → 一次forward → [a1,a2,a3,a4]
flit3 → ZMQ → forward → a3    flit3─┘
flit4 → ZMQ → forward → a4
 4次往返 × 5ms = 20ms          1次往返 × 8ms = 8ms
```

### 实现要点

**C++ 侧 (adqn_routing.cc)**：
```cpp
// 新增批量请求队列
static std::vector<BatchRequest> g_batch_queue;  // 最多攒 BATCH_SIZE 个
static int g_batch_counter = 0;

int outportComputeAdqn(...) {
    // 构造tokens
    float tokens[3][18];
    buildTokenStates(router, route, inport_dirn, tokens);
    int action_mask[4];
    buildActionMask(router, route, action_mask);
    
    // 加入批量队列
    BatchRequest req = {router_id, tokens, action_mask};
    g_batch_queue.push_back(req);
    g_batch_counter++;
    
    if (g_batch_counter >= BATCH_SIZE) {
        // 发送整个batch
        sendBatchZMQ(g_batch_queue);
        // 接收batch结果
        std::vector<int> actions = recvBatchZMQ();
        // 分发结果
        for (int i = 0; i < BATCH_SIZE; i++)
            g_batch_queue[i].result = actions[i];
    }
    
    // 等待本请求的结果（可能需要轮询）
    return getMyResult();
}
```

**Python 侧 (madrl_server.py)**：
```python
def decision_thread_batch(ctx, agent, port, running):
    s = ctx.socket(zmq.REP); s.bind(f"tcp://*:{port}")
    while running[0]:
        msg = s.recv()
        # 解析batch: [batch_size, float*tokens*bs, int*mask*bs]
        batch_size = struct.unpack('i', msg[:4])[0]
        states = []; masks = []
        offset = 4
        for _ in range(batch_size):
            tk = np.array(struct.unpack(f'{fc}f', msg[offset:offset+fc*4]))
            states.append(tk.reshape(NUM_TOKENS, TOKEN_DIM))
            offset += fc*4
            am = np.array(struct.unpack(f'{NUM_ACTIONS}i', msg[offset:offset+NUM_ACTIONS*4]))
            masks.append(am)
            offset += NUM_ACTIONS*4
        
        # 一次forward处理整个batch
        st = torch.FloatTensor(np.array(states))
        mt = torch.FloatTensor(np.array(masks))
        with torch.no_grad():
            _, pr, _ = agent.net(st, mt, nb_as=None)
            actions = pr.argmax(dim=1).cpu().numpy()
        
        # 返回batch结果
        s.send(struct.pack(f'{batch_size}i', *actions))
```

### 效果预估
- ZMQ往返次数减少 N 倍（N=batch_size）
- 决策延迟降低 ~60%
- 吞吐量可支撑到 0.30+

### 风险
- 批量等待增加单个flit延迟（攒batch的时间）
- 需要协调多个flit的异步结果返回

---

## 方案二：C++ LibTorch 本地推理

### 思路
将训练好的PyTorch模型导出为TorchScript，在gem5 C++侧直接加载推理，完全消除ZMQ。

### 实现步骤

**步骤1：导出模型**
```python
# 在Python侧导出
agent.net.eval()
example_input = torch.randn(1, 3, 18)
example_mask = torch.ones(1, 4)
traced = torch.jit.trace(agent.net, (example_input, example_mask))
traced.save("adqn_router.pt")
```

**步骤2：gem5集成LibTorch**
```cpp
// adqn_routing.cc
#include <torch/script.h>

static torch::jit::script::Module g_model;

void initAdqnModel(const std::string& model_path) {
    g_model = torch::jit::load(model_path);
    g_model.eval();
}

int outportComputeAdqn(Router* router, ...) {
    float tokens[3][18];
    buildTokenStates(router, route, inport_dirn, tokens);
    
    // 直接本地推理，无ZMQ
    torch::Tensor st = torch::from_blob(tokens, {1, 3, 18});
    torch::Tensor mt = torch::ones({1, 4});
    auto outputs = g_model.forward({st, mt});
    auto probs = torch::softmax(outputs[1], -1);  // masked probs
    int action = probs.argmax().item<int>();
    return actionToOutport(action, router);
}
```

### 效果预估
- ZMQ延迟完全消除（0 → 本地推理 ~0.1ms）
- 仿真时间缩短 95%+
- 吞吐量可达XY级别

### 风险
- LibTorch C++依赖（~2GB），增加编译复杂度
- TorchScript不支持所有PyTorch操作（需验证BatchNorm/LayerNorm兼容性）
- 模型更新需要重新导出

---

## 方案三：自适应 Toward-Restriction

### 思路
当前toward-restriction是硬约束（只能向目标方向走）。改为软约束：当所选方向严重拥塞时，允许向非目标方向绕路。

### 实现

**C++ 侧 (adqn_routing.cc)**：
```cpp
void buildActionMask(Router* router, RouteInfo route, int action_mask[4]) {
    // 默认：只允许向目标方向
    // 但当所选方向 credit < THRESHOLD 时，开放所有方向
    
    float min_credit_toward = getTowardMinCredit(router, route);
    
    if (min_credit_toward < 0.2) {  // 向目标方向严重拥塞
        // 开放所有非边界方向（允许绕路）
        for (int d = 0; d < 4; d++)
            action_mask[d] = isDirectionValid(router, d) ? 1 : 0;
    } else {
        // 标准toward-restriction
        towardOnly(router, route, action_mask);
    }
}
```

**奖励函数调整**：
```cpp
float computeReward(int action, Router* router, RouteInfo route) {
    // ... existing code ...
    
    // 新增：绕路惩罚（但比完全阻塞好）
    if (!isTowardDestination(action, router, route)) {
        detour_penalty = 0.15;  // 轻惩罚，允许学习绕路
    }
    
    return base_reward + congestion_term + coordination_bonus 
           - global_penalty - detour_penalty;
}
```

### 效果预估
- 0.24+ 拥塞时可通过绕路避开hotspot
- 投递率从 94.6% 恢复到 99%+
- 延迟可能略有增加（绕路多跳），但吞吐量大幅改善

### 风险
- 动作空间增大，需要更多训练
- 可能导致活锁（flit无限绕路）→ 需加入最大跳数限制

---

## 方案四：混合路由（XY fallback）

### 思路
ADQN在高负载下fallback到XY路由。XY虽不感知拥塞，但速度快、无ZMQ延迟。

### 实现

```cpp
int outportComputeAdqn(Router* router, ...) {
    // 检测拥塞程度
    float congestion_level = router->getCongestionLevel();
    
    if (congestion_level > 0.8 || zmq_latency > threshold) {
        // Fallback到XY
        return xyRoute(router, dest_id);
    }
    
    // 正常ADQN决策
    return adqnRoute(router, ...);
}
```

### 效果预估
- 低负载：ADQN自适应优势保持
- 高负载：XY保证基本吞吐量
- 综合性能优于纯ADQN或纯XY

---

## 方案五：减少网络复杂度（立即可做）

### 思路
当前网络每个forward包含：BN → Embed → PE → 2×CrossAttn(8head) → SharedFC → Actor+Critic+Fusion。减少层数/头数加速推理。

### 建议调整

| 参数 | 当前值 | 建议值 | 推理加速 |
|------|--------|--------|---------|
| num_attn_layers | 2 | 1 | ~40% |
| num_heads | 8 | 4 | ~30% |
| embed_dim | 64 | 48 | ~25% |
| ff_dim | 128 | 96 | ~20% |

**组合效果**：推理速度提升约 2-3 倍。

### 风险
- 模型容量降低，可能影响决策质量
- 需要重新训练验证

---

## 优先级建议

| 优先级 | 方案 | 实现难度 | 预期效果 | 时间 |
|--------|------|---------|---------|------|
| **P0** | 方案五：简化网络 | 低（改几个参数） | 推理提速2-3x | 30分钟 |
| **P0** | 方案三：自适应Toward | 中（改C++ mask+reward） | 解决0.24+崩溃 | 1小时 |
| **P1** | 方案一：批量推理 | 中（改ZMQ协议） | ZMQ延迟降60%+ | 2-3小时 |
| **P2** | 方案四：XY fallback | 低（改C++） | 高负载保底 | 30分钟 |
| **P3** | 方案二：LibTorch | 高（编译+导出） | 完全消除ZMQ | 1-2天 |

---

## 快速验证计划

1. **先做方案五**：减少CrossAttn层数 2→1，head数 8→4，重新训练一个注入率（0.20）看延迟和推理速度
2. **再做方案三**：开放toward-restriction的软约束，在0.24训练验证吞吐量恢复
3. **如仍不够**：实施方案一（批量推理）或方案二（LibTorch）
