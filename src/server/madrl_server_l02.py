#!/usr/bin/env python3
"""MLP Smart + Q-transfer (λ=0.2): previous-step direction-specific Q-value sharing."""
import argparse, signal, struct, threading, time, random, math, os
from collections import deque
import numpy as np
try: import zmq
except ImportError: print("Error: pyzmq not installed."); raise
try: import torch; import torch.nn as nn; import torch.nn.functional as F
except ImportError: print("Error: PyTorch not installed."); raise

NUM_TOKENS=3; TOKEN_DIM=18; NUM_ACTIONS=4
DIM_RX=0; DIM_RY=1; DIM_DX=2; DIM_DY=3; DIM_HOPS=4; DIM_INPORT=13
REPLAY_CAPACITY=200000; BATCH_SIZE=128; GAMMA=0.99; LR=1e-3
EPS_START=1.0; EPS_END=0.05; EPS_DECAY=0.9995; TARGET_UPDATE=500
LAMBDA=0.04
DIR_ENCODE={0.0:"East",0.25:"West",0.5:"North",0.75:"South"}
NUM_COLS=8; NUM_ROWS=8

# ======================== MLP Network ========================
class MLPQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(TOKEN_DIM, track_running_stats=True)
        self.fc1 = nn.Linear(NUM_TOKENS*TOKEN_DIM, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, NUM_ACTIONS)
    def forward(self, x, mask=None):
        B, T, D = x.shape[0], NUM_TOKENS, TOKEN_DIM
        xf = x.reshape(-1, D) if x.dim()==3 else x; xf = self.bn(xf)
        x = xf.reshape(B, T*D); x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)); q = self.fc3(x)
        if mask is not None: q = q.masked_fill(mask==0, float('-inf'))
        return q

# ======================== State Buffer for Q-transfer ========================
class StateBuffer:
    def __init__(self):
        self.lock = threading.Lock(); self.prev_q = {}  # rid→[4]
        self.q_sum = {}; self.q_count = {}  # for warm-start export
    def update(self, rid, q4):
        with self.lock:
            self.prev_q[rid] = q4.copy()
            if rid not in self.q_sum:
                self.q_sum[rid] = np.zeros(4, dtype=np.float64)
                self.q_count[rid] = 0
            self.q_sum[rid] += q4.astype(np.float64)
            self.q_count[rid] += 1
    def get_downstream(self, rid, t0, cols=NUM_COLS, rows=NUM_ROWS):
        x=int(round(t0[DIM_RX])); y=int(round(t0[DIM_RY]))
        dx=int(round(t0[DIM_DX])); dy=int(round(t0[DIM_DY]))
        qd = np.zeros(4, dtype=np.float32)
        with self.lock:
            if dx>x and (rid+1) in self.prev_q: qd = self.prev_q[rid+1].copy()
            elif dx<x and (rid-1) in self.prev_q: qd = self.prev_q[rid-1].copy()
            if dy>y and (rid+cols) in self.prev_q: qd = self.prev_q[rid+cols].copy()
            elif dy<y and (rid-cols) in self.prev_q: qd = self.prev_q[rid-cols].copy()
        return qd

# ======================== Replay Buffer ========================
class ReplayBuffer:
    def __init__(self, c=REPLAY_CAPACITY): self.b = deque(maxlen=c)
    def push(self, s, a, r, ns, qd, qdn, t=False):
        self.b.append((s, a, r, ns, qd, qdn, t))
    def sample(self, n=BATCH_SIZE):
        bt = random.sample(self.b, min(n, len(self.b)))
        ss,aa,rr,nss,qds,qdn,ts = zip(*bt)
        return (torch.FloatTensor(np.array(ss)), torch.LongTensor(aa),
                torch.FloatTensor(rr), torch.FloatTensor(np.array(nss)),
                torch.FloatTensor(np.array(qds)), torch.FloatTensor(np.array(qdn)),
                torch.BoolTensor(ts))
    def __len__(self): return len(self.b)

# ======================== Agent ========================
class DQNAgent:
    def __init__(self):
        self.net=MLPQNet(); self.target=MLPQNet()
        self.target.load_state_dict(self.net.state_dict())
        self.opt=torch.optim.Adam(self.net.parameters(), lr=LR)
        self.replay=ReplayBuffer(); self.steps=0; self.epsilon=EPS_START
        self.lh=[]; self.rh=[]; self.rb=[]
        self.pending={}; self.pl=threading.Lock()
        self.sbuf=StateBuffer(); self.qd_cache={}

    def select_action(self, s, am, rid, exploit=False):
        if am.sum()==0: return 0
        if not exploit and random.random()<self.epsilon:
            valid=np.where(am>0)[0]; return int(random.choice(valid))
        with torch.no_grad():
            st=torch.FloatTensor(s).unsqueeze(0); mt=torch.FloatTensor(am).unsqueeze(0)
            q_local=self.net(st,mt).squeeze(0).cpu().numpy()
            q_down=self.sbuf.get_downstream(rid, s[0])
            q_aug=q_local + LAMBDA*q_down
            q_masked=np.where(am>0,q_aug,-1e9)
            return int(np.argmax(q_masked))

    def log_reward(self,r): self.rb.append(r)
    def flush_rewards(self):
        if self.rb: self.rh.append((self.steps,sum(self.rb)/len(self.rb))); self.rb=[]

    def try_complete(self, s, am, rid):
        s0=s[0]; mh=NUM_ROWS+NUM_COLS-2 or 1
        ip=DIR_ENCODE.get(min([0.0,0.25,0.5,0.75,1.0],key=lambda c:abs(s0[DIM_INPORT]-c)),"Local")
        x,y,dx,dy=[int(round(s0[i])) for i in [0,1,2,3]]; ht=int(round(s0[DIM_HOPS]*mh))
        px,py=x,y
        if ip=="East": px=x-1
        elif ip=="West": px=x+1
        elif ip=="North": py=y-1
        elif ip=="South": py=y+1
        pk=(px,py,dx,dy,max(0,ht-1))
        with self.pl:
            if pk in self.pending:
                ps,pa,pr,pqd=self.pending.pop(pk)
                qd_cur=self.sbuf.get_downstream(rid,s0)
                self.replay.push(ps,pa,pr,s.copy(),pqd,qd_cur,False); return True
        return False

    def buffer_exp(self, s, a, r, q_down=None):
        if q_down is None: q_down=np.zeros(4,dtype=np.float32)
        s0=s[0]; mh=NUM_ROWS+NUM_COLS-2 or 1
        k=(int(round(s0[0])),int(round(s0[1])),int(round(s0[2])),
           int(round(s0[3])),int(round(s0[DIM_HOPS]*mh)))
        with self.pl:
            self.pending[k]=(s.copy(),a,r,q_down)
            if len(self.pending)>10000:
                ks=list(self.pending.keys())
                for kd in ks[:len(ks)//2]: self.pending.pop(kd,None)

    def train_step(self):
        self.flush_rewards()
        if len(self.replay)<BATCH_SIZE: return
        ss,aa,rr,nss,qds,qdn,ts=self.replay.sample()
        q_aug=self.net(ss)+LAMBDA*qds
        q_cur=q_aug.gather(1,aa.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq_aug=self.target(nss)+LAMBDA*qdn; nq=nq_aug.max(1)[0]
            target=rr+GAMMA*nq*(~ts)
        loss=F.smooth_l1_loss(q_cur,target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.lh.append((self.steps,loss.item()))
        if self.steps%TARGET_UPDATE==0: self.target.load_state_dict(self.net.state_dict())
        if self.epsilon>EPS_END: self.epsilon=max(EPS_END,self.epsilon*EPS_DECAY)

# ======================== Threads ========================
def decision_thread(ctx,agent,port,running):
    s=ctx.socket(zmq.REP); s.bind(f"tcp://*:{port}")
    fc=NUM_TOKENS*TOKEN_DIM
    while running[0]:
        try: msg=s.recv(zmq.NOBLOCK if not running[0] else 0)
        except zmq.ZMQError:
            if running[0]: time.sleep(0.001); continue
        if len(msg)!=fc*4+4*4+4: s.send(struct.pack('i',0)); continue
        tk=np.array(struct.unpack(f'{fc}f',msg[:fc*4])).reshape(NUM_TOKENS,TOKEN_DIM)
        am=np.array(struct.unpack('4i',msg[fc*4:fc*4+16])).astype(int)
        rid=struct.unpack('i',msg[fc*4+16:])[0]
        agent.try_complete(tk,am,rid)
        # Cache q_down for experience thread
        qd=agent.sbuf.get_downstream(rid,tk[0])
        a=agent.select_action(tk,am,rid)
        # Cache q_local for state buffer
        with torch.no_grad():
            st=torch.FloatTensor(tk).unsqueeze(0)
            ql=agent.net(st).squeeze(0).cpu().numpy()
            agent.sbuf.update(rid,ql)
        mh=NUM_ROWS+NUM_COLS-2 or 1
        fk=(int(round(tk[0][0])),int(round(tk[0][1])),int(round(tk[0][2])),
            int(round(tk[0][3])),int(round(tk[0][DIM_HOPS]*mh)))
        agent.qd_cache[fk]=qd
        s.send(struct.pack('i',a))

def experience_thread(ctx,agent,port,running):
    s=ctx.socket(zmq.PULL); s.bind(f"tcp://*:{port}")
    p=zmq.Poller(); p.register(s,zmq.POLLIN)
    fc=NUM_TOKENS*TOKEN_DIM
    while running[0]:
        if not dict(p.poll(100)): continue
        try: msg=s.recv(zmq.NOBLOCK)
        except: continue
        if len(msg)!=fc*4+4+4+1: continue
        tk=np.array(struct.unpack(f'{fc}f',msg[:fc*4])).reshape(NUM_TOKENS,TOKEN_DIM)
        a=struct.unpack('i',msg[fc*4:fc*4+4])[0]
        r=struct.unpack('f',msg[fc*4+4:fc*4+8])[0]
        term=msg[fc*4+8]!=0
        agent.log_reward(r)
        mh=NUM_ROWS+NUM_COLS-2 or 1
        fk=(int(round(tk[0][0])),int(round(tk[0][1])),int(round(tk[0][2])),
            int(round(tk[0][3])),int(round(tk[0][DIM_HOPS]*mh)))
        qd=agent.qd_cache.pop(fk,np.zeros(4,dtype=np.float32))
        if term:
            agent.replay.push(tk,a,r,
                np.zeros((NUM_TOKENS,TOKEN_DIM),dtype=np.float32),
                qd,np.zeros(4,dtype=np.float32),True)
        else:
            agent.buffer_exp(tk,a,r,qd)
        agent.steps+=1
        if agent.steps%1000==0:
            print(f"[exp] s={agent.steps} buf={len(agent.replay)} pend={len(agent.pending)} eps={agent.epsilon:.4f}")

def training_thread(agent,running):
    while running[0]:
        if len(agent.replay)>=BATCH_SIZE: agent.train_step()
        time.sleep(0.01)

def export_warm_q(agent, path):
    """Export latest q_local per router from StateBuffer as warm-start Q-buffer."""
    lines = ["// Warm-start Q-buffer: latest q_local per router",
             f"// λ={LAMBDA} routers={len(agent.sbuf.prev_q)}",""]
    for rid in sorted(agent.sbuf.prev_q.keys()):
        q = agent.sbuf.prev_q[rid]
        lines.append(f"static const float warm_q_{rid}[4] = {{{q[0]:.6f}f, {q[1]:.6f}f, {q[2]:.6f}f, {q[3]:.6f}f}};")
    with open(path, 'w') as f: f.write("\n".join(lines))
    print(f"[export] warm Q written to {path} ({len(agent.sbuf.prev_q)} routers)")

def export_weights(agent,path):
    sd=agent.net.state_dict()
    lines=["// MLP+Q-transfer weights","",f"// λ={LAMBDA} tokens={NUM_TOKENS} dim={TOKEN_DIM}",""]
    for name,param in sd.items():
        arr=param.cpu().numpy(); flat=arr.flatten()
        shape="*".join(str(d) for d in arr.shape)
        lines.append(f"// {name} [{shape}]")
        lines.append(f"static const float {name.replace('.','_')}[{len(flat)}] = {{")
        for i in range(0,len(flat),8):
            lines.append("  "+",".join(f"{v:.8f}f" for v in flat[i:i+8])+",")
        lines.append("};")
        lines.append("")
    with open(path,'w') as f: f.write("\n".join(lines))
    print(f"[export] written to {path}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--decision-port',type=int,default=7777)
    parser.add_argument('--experience-port',type=int,default=7778)
    parser.add_argument('--checkpoint',type=str,default=None)
    parser.add_argument('--save',type=str,default='mlp_l02.pt')
    parser.add_argument('--save-interval',type=int,default=10000)
    parser.add_argument('--mesh-cols',type=int,default=8)
    parser.add_argument('--mesh-rows',type=int,default=8)
    parser.add_argument('--epsilon',type=float,default=-1.0)
    parser.add_argument('--inj-rate',type=float,default=0.1)
    parser.add_argument('--export',type=str,default=None)
    args=parser.parse_args()
    global NUM_COLS,NUM_ROWS; NUM_COLS=args.mesh_cols; NUM_ROWS=args.mesh_rows
    print("="*60)
    print(f"MLP Smart + Q-transfer λ={LAMBDA} | {NUM_TOKENS}tok×{TOKEN_DIM}D | FC 54→32→16→4")
    print(f"  Mesh: {NUM_COLS}x{NUM_ROWS} | Rate: {args.inj_rate}")
    print("="*60)
    agent=DQNAgent()
    if args.checkpoint:
        print(f"[init] loading {args.checkpoint}")
        sd=torch.load(args.checkpoint,map_location='cpu'); md=agent.net.state_dict()
        fl={k:v for k,v in sd.items() if k in md and md[k].shape==v.shape}
        md.update(fl); agent.net.load_state_dict(md)
        if args.epsilon<0: agent.epsilon=EPS_END
        print(f"[init] loaded {len(fl)}/{len(md)} params")
    if args.epsilon>=0: agent.epsilon=args.epsilon
    inf=args.epsilon==0.0
    ctx=zmq.Context(); running=[True]; sv=[args.save]
    def do_save(sfx=""):
        p=sv[0].replace('.pt',f'{sfx}.pt') if sfx else sv[0]
        print(f"[save] {p}"); torch.save(agent.net.state_dict(),p)
    signal.signal(signal.SIGTERM,lambda *_:running.__setitem__(0,False))
    signal.signal(signal.SIGINT,lambda *_:running.__setitem__(0,False))
    ts=[threading.Thread(target=decision_thread,args=(ctx,agent,args.decision_port,running),daemon=True)]
    if not inf:
        ts.append(threading.Thread(target=experience_thread,args=(ctx,agent,args.experience_port,running),daemon=True))
        ts.append(threading.Thread(target=training_thread,args=(agent,running),daemon=True))
    for t in ts: t.start()
    print("[server] Inference" if inf else "[server] Training")
    ls=0
    while running[0]:
        time.sleep(5)
        if args.save_interval>0 and agent.steps-ls>=args.save_interval:
            do_save(f"_step{agent.steps}"); ls=agent.steps
    do_save()
    export_weights(agent,"/home/nayomi/gem5/adqn_results/mlp_weights.h")
    export_warm_q(agent,"/home/nayomi/gem5/adqn_results/warm_q.h")
    if args.export: export_weights(agent,args.export)
    ctx.term(); print("[server] Done.")

if __name__=='__main__': main()
