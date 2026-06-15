/*
 * Copyright (c) 2024
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __MEM_RUBY_NETWORK_GARNET_ZMQ_FUNCTIONS_HH__
#define __MEM_RUBY_NETWORK_GARNET_ZMQ_FUNCTIONS_HH__

#include "mem/ruby/network/garnet/adqn_routing.hh"

namespace gem5
{

namespace ruby
{

namespace garnet
{

#ifdef HAVE_ZMQ

/**
 * Initialize ZMQ context and sockets.
 * Must be called once before any zmq_send_* or zmq_recv_* calls.
 * Lazily connects on first call; safe to call multiple times.
 *
 * @param decision_port   Port for ZMQ REQ socket (inference)
 * @param experience_port Port for ZMQ PUSH socket (experience replay)
 */
void zmq_init(int decision_port, int experience_port);

/**
 * Send inference request to Python server and receive action.
 * ATQ-Route: sends 1-token (18D) state + action_mask + router_id.
 *
 * @param tokens      1×18 state array (current router only)
 * @param action_mask 4-element mask (1=valid, 0=blocked)
 * @param router_id   Router ID for downstream Q-lookup in Python
 * @return Action index 0..3 (E, W, N, S)
 */
int zmq_send_inference(const float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
                       const int action_mask[ADQN_NUM_ACTIONS],
                       int router_id,
                       const int toward_ids[ADQN_NUM_ACTIONS],
                       class GarnetNetwork *net);

// Batch request: one entry in the pending queue
struct BatchRequest {
    int router_id;              // Which router (for result distribution)
    float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM];
    int action_mask[ADQN_NUM_ACTIONS];
    int result;                 // -1 = pending, 0-3 = result
    float reward;               // Computed after action
    bool terminal;              // Terminal flag
};

// Batch inference globals
constexpr int ADQN_BATCH_SIZE = 4;
extern int g_adqn_batch_seq;   // Global sequence counter for batch flush

/**
 * Add a routing request to the batch queue.
 * When the queue reaches ADQN_BATCH_SIZE, sends all requests to Python
 * in a single ZMQ round-trip and distributes results.
 * Returns the action for THIS request.
 *
 * @param router_id   Router ID for result tracking
 * @param tokens      3×18 state array
 * @param action_mask 4-element mask
 * @param out_reward  [out] reward computed by C++ after getting action
 * @param out_terminal [out] whether this hop reaches destination
 * @return Action index 0..3
 */
int zmq_send_inference_batch(int router_id,
                             const float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
                             const int action_mask[ADQN_NUM_ACTIONS],
                             float &out_reward, bool &out_terminal);

/**
 * Flush any remaining requests in the batch queue (send even if not full).
 */
void zmq_flush_batch();

/**
 * Send experience transition to Python DQN server for replay buffer.
 * Non-blocking fire-and-forget.
 *
 * @param tokens   3×18 state array that produced the action
 * @param action   Chosen action 0..3
 * @param reward   Immediate reward for the action
 * @param terminal True if this action leads directly to the destination
 */
void zmq_send_experience(const float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
                         int action, float reward, bool terminal);

/**
 * Cleanup ZMQ resources. Called at simulation end.
 */
void zmq_cleanup();

#else

// Stub: when ZMQ is not available, panic if ADQN routing is attempted
static inline void zmq_init(int, int) {
    panic("ADQN routing requires libzmq. Please install libzmq-dev and rebuild.");
}
static inline int zmq_send_inference(const float[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
                                     const int[ADQN_NUM_ACTIONS]) {
    panic("ADQN routing requires libzmq.");
    return 0;
}
static inline void zmq_send_experience(const float[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
                                       int, float, bool) {
    panic("ADQN routing requires libzmq.");
}
static inline void zmq_cleanup() {}

#endif // HAVE_ZMQ

} // namespace garnet
} // namespace ruby
} // namespace gem5

#endif // __MEM_RUBY_NETWORK_GARNET_ZMQ_FUNCTIONS_HH__
