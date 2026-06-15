#!/usr/bin/env python3
"""MLP Smart DQN Server: FC(54→32→16→4) + smart detour mask + credit-scaled reward."""
import argparse, signal, struct, threading, time, random, math, os
from collections import deque
import numpy as np
try: import zmq
except ImportError: print("Error: pyzmq not installed."); raise
try: import torch; import torch.nn as nn; import torch.nn.functional as F
except ImportError: print("Error: PyTorch not installed."); raise

NUM_TOKENS=3; TOKEN_DIM=18; NUM_ACTIONS=4
DIM_RX_NORM=0; DIM_RY_NORM=1; DIM_DX_NORM=2; DIM_DY_NORM=3
DIM_HOPS_TAKEN=4; DIM_INPORT_DIR=13
REPLAY_CAPACITY=200000; BATCH_SIZE=128; GAMMA=0.99; LR=1e-3
EPS_START=1.0; EPS_END=0.05; EPS_DECAY=0.9995; TARGET_UPDATE=500
DIR_ENCODE={0.0:"East",0.25:"West",0.5:"North",0.75:"South"}
NUM_COLS=8; NUM_ROWS=8

class MLPQNet(nn.Module):
    """Simple MLP: BN per-token → flatten 54D → 32 → 16 → 4 Q-values."""
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(TOKEN_DIM, track_running_stats=True)
        self.fc1 = nn.Linear(NUM_TOKENS * TOKEN_DIM, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, NUM_ACTIONS)
    def forward(self, x, mask=None):
        B, T, D = x.shape[0], NUM_TOKENS, TOKEN_DIM
        xf = x.reshape(-1, D) if x.dim() == 3 else x
        xf = self.bn(xf); x = xf.reshape(B, T * D)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); q = self.fc3(x)
        if mask is not None: q = q.masked_fill(mask == 0, float('-inf'))
        return q

class ReplayBuffer:
    def __init__(self, c=REPLAY_CAPACITY): self.b = deque(maxlen=c)
    def push(self, s, a, r, ns, t=False): self.b.append((s, a, r, ns, t))
    def sample(self, n=BATCH_SIZE):
        bt = random.sample(self.b, min(n, len(self.b)))
        ss, aa, rr, nss, ts = zip(*bt)
        return (torch.FloatTensor(np.array(ss)), torch.LongTensor(aa),
                torch.FloatTensor(rr), torch.FloatTensor(np.array(nss)),
                torch.BoolTensor(ts))
    def __len__(self): return len(self.b)

class DQNAgent:
    def __init__(self):
        self.net = MLPQNet(); self.target = MLPQNet()
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.replay = ReplayBuffer(); self.steps = 0; self.epsilon = EPS_START
        self.lh = []; self.rh = []; self.rb = []
        self.pending = {}; self.pl = threading.Lock()
    def select_action(self, s, am, rid, exploit=False):
        if am.sum() == 0: return 0
        if not exploit and random.random() < self.epsilon:
            valid = np.where(am > 0)[0]; return int(random.choice(valid))
        with torch.no_grad():
            st = torch.FloatTensor(s).unsqueeze(0); mt = torch.FloatTensor(am).unsqueeze(0)
            return int(np.argmax(self.net(st, mt).squeeze(0).cpu().numpy()))
    def log_reward(self, r): self.rb.append(r)
    def flush_rewards(self):
        if self.rb: self.rh.append((self.steps, sum(self.rb)/len(self.rb))); self.rb = []
    def try_complete(self, s, am):
        s0 = s[0]; mh = NUM_ROWS+NUM_COLS-2 or 1
        ip = DIR_ENCODE.get(min([0.0,0.25,0.5,0.75,1.0], key=lambda c: abs(s0[13]-c)), "Local")
        x,y,dx,dy = [int(round(s0[i])) for i in [0,1,2,3]]; ht = int(round(s0[4]*mh))
        px, py = x, y
        if ip == "East": px = x-1
        elif ip == "West": px = x+1
        elif ip == "North": py = y-1
        elif ip == "South": py = y+1
        pk = (px, py, dx, dy, max(0, ht-1))
        with self.pl:
            if pk in self.pending:
                ps, pa, pr = self.pending.pop(pk)
                self.replay.push(ps, pa, pr, s.copy(), False); return True
        return False
    def buffer_exp(self, s, a, r):
        s0 = s[0]; mh = NUM_ROWS+NUM_COLS-2 or 1
        k = (int(round(s0[0])), int(round(s0[1])), int(round(s0[2])),
             int(round(s0[3])), int(round(s0[4]*mh)))
        with self.pl:
            self.pending[k] = (s.copy(), a, r)
            if len(self.pending) > 10000:
                ks = list(self.pending.keys())
                for kd in ks[:len(ks)//2]: self.pending.pop(kd, None)
    def train_step(self):
        self.flush_rewards()
        if len(self.replay) < BATCH_SIZE: return
        ss, aa, rr, nss, ts = self.replay.sample()
        q = self.net(ss).gather(1, aa.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.net(nss).max(1)[0]; target = rr + GAMMA * nq * (~ts)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.lh.append((self.steps, loss.item()))
        if self.steps % TARGET_UPDATE == 0: self.target.load_state_dict(self.net.state_dict())
        if self.epsilon > EPS_END: self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

def decision_thread(ctx, agent, port, running):
    s = ctx.socket(zmq.REP); s.bind(f"tcp://*:{port}")
    fc = NUM_TOKENS * TOKEN_DIM
    while running[0]:
        try: msg = s.recv(zmq.NOBLOCK if not running[0] else 0)
        except zmq.ZMQError:
            if running[0]: time.sleep(0.001); continue
        if len(msg) != fc*4 + 4*4 + 4: s.send(struct.pack('i', 0)); continue
        tk = np.array(struct.unpack(f'{fc}f', msg[:fc*4])).reshape(NUM_TOKENS, TOKEN_DIM)
        am = np.array(struct.unpack('4i', msg[fc*4:fc*4+16])).astype(int)
        rid = struct.unpack('i', msg[fc*4+16:])[0]
        agent.try_complete(tk, am); a = agent.select_action(tk, am, rid)
        s.send(struct.pack('i', a))

def experience_thread(ctx, agent, port, running):
    s = ctx.socket(zmq.PULL); s.bind(f"tcp://*:{port}")
    p = zmq.Poller(); p.register(s, zmq.POLLIN)
    fc = NUM_TOKENS * TOKEN_DIM
    while running[0]:
        if not dict(p.poll(100)): continue
        try: msg = s.recv(zmq.NOBLOCK)
        except: continue
        if len(msg) != fc*4 + 4 + 4 + 1: continue
        tk = np.array(struct.unpack(f'{fc}f', msg[:fc*4])).reshape(NUM_TOKENS, TOKEN_DIM)
        a = struct.unpack('i', msg[fc*4:fc*4+4])[0]
        r = struct.unpack('f', msg[fc*4+4:fc*4+8])[0]
        term = msg[fc*4+8] != 0
        agent.log_reward(r)
        if term: agent.replay.push(tk, a, r, np.zeros((NUM_TOKENS, TOKEN_DIM), dtype=np.float32), True)
        else: agent.buffer_exp(tk, a, r)
        agent.steps += 1
        if agent.steps % 1000 == 0:
            print(f"[exp] s={agent.steps} buf={len(agent.replay)} pend={len(agent.pending)} eps={agent.epsilon:.4f}")

def training_thread(agent, running):
    while running[0]:
        if len(agent.replay) >= BATCH_SIZE: agent.train_step()
        time.sleep(0.01)

def export_weights(agent, path):
    sd = agent.net.state_dict()
    lines = ["// MLP Smart weights for gem5 native inference",
             f"// tokens={NUM_TOKENS} dim={TOKEN_DIM} hidden=32,16 actions={NUM_ACTIONS}", ""]
    for name, param in sd.items():
        arr = param.cpu().numpy(); flat = arr.flatten()
        shape = "*".join(str(d) for d in arr.shape)
        lines.append(f"// {name} [{shape}]")
        lines.append(f"static const float {name.replace('.','_')}[{len(flat)}] = {{")
        for i in range(0, len(flat), 8):
            lines.append("  " + ",".join(f"{v:.8f}f" for v in flat[i:i+8]) + ",")
        lines.append("};")
        lines.append("")
    with open(path, 'w') as f: f.write("\n".join(lines))
    print(f"[export] written to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision-port', type=int, default=7777)
    parser.add_argument('--experience-port', type=int, default=7778)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save', type=str, default='mlp_smart.pt')
    parser.add_argument('--save-interval', type=int, default=10000)
    parser.add_argument('--mesh-cols', type=int, default=8)
    parser.add_argument('--mesh-rows', type=int, default=8)
    parser.add_argument('--epsilon', type=float, default=-1.0)
    parser.add_argument('--inj-rate', type=float, default=0.1)
    parser.add_argument('--export', type=str, default=None)
    args = parser.parse_args()
    global NUM_COLS, NUM_ROWS; NUM_COLS = args.mesh_cols; NUM_ROWS = args.mesh_rows
    print("="*60)
    print(f"MLP Smart DQN | {NUM_TOKENS} tok x {TOKEN_DIM}D | FC 54→32→16→4")
    print(f"  Mesh: {NUM_COLS}x{NUM_ROWS} | Rate: {args.inj_rate}")
    print("="*60)
    agent = DQNAgent()
    if args.checkpoint:
        print(f"[init] loading {args.checkpoint}")
        sd = torch.load(args.checkpoint, map_location='cpu'); md = agent.net.state_dict()
        fl = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
        md.update(fl); agent.net.load_state_dict(md)
        if args.epsilon < 0: agent.epsilon = EPS_END
        print(f"[init] loaded {len(fl)}/{len(md)} params")
    if args.epsilon >= 0: agent.epsilon = args.epsilon
    inf = args.epsilon == 0.0
    ctx = zmq.Context(); running = [True]; sv = [args.save]
    def do_save(sfx=""):
        p = sv[0].replace('.pt', f'{sfx}.pt') if sfx else sv[0]
        print(f"[save] {p}"); torch.save(agent.net.state_dict(), p)
    signal.signal(signal.SIGTERM, lambda *_: running.__setitem__(0, False))
    signal.signal(signal.SIGINT, lambda *_: running.__setitem__(0, False))
    ts = [threading.Thread(target=decision_thread, args=(ctx, agent, args.decision_port, running), daemon=True)]
    if not inf:
        ts.append(threading.Thread(target=experience_thread, args=(ctx, agent, args.experience_port, running), daemon=True))
        ts.append(threading.Thread(target=training_thread, args=(agent, running), daemon=True))
    for t in ts: t.start()
    print("[server] Inference" if inf else "[server] Training")
    ls = 0
    while running[0]:
        time.sleep(5)
        if args.save_interval > 0 and agent.steps - ls >= args.save_interval:
            do_save(f"_step{agent.steps}"); ls = agent.steps
    do_save()
    export_weights(agent, "/home/nayomi/gem5/adqn_results/mlp_weights.h")
    if args.export: export_weights(agent, args.export)
    ctx.term(); print("[server] Done.")

if __name__ == '__main__': main()
