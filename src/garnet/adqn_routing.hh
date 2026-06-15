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

#ifndef __MEM_RUBY_NETWORK_GARNET_ADQN_ROUTING_HH__
#define __MEM_RUBY_NETWORK_GARNET_ADQN_ROUTING_HH__

#include "mem/ruby/network/garnet/CommonTypes.hh"

namespace gem5
{

namespace ruby
{

namespace garnet
{

class Router;
class GarnetNetwork;

// ATQ-Route Dueling DQN: 3 tokens (self + 2 toward neighbors)
constexpr int ADQN_NUM_TOKENS = 3;
// Number of features per token
constexpr int ADQN_TOKEN_DIM = 18;
// Number of actions: E=0, W=1, N=2, S=3
constexpr int ADQN_NUM_ACTIONS = 4;

// Max credits per VC for normalization (buffers_per_data_vc)
constexpr float MAX_CREDITS = 4.0f;

// Token indices
constexpr int TOKEN_CURRENT  = 0;
constexpr int TOKEN_TOWARD1  = 1;
constexpr int TOKEN_TOWARD2  = 2;

// Feature dimension indices (14D per token)
constexpr int DIM_RX_NORM         = 0;
constexpr int DIM_RY_NORM         = 1;
constexpr int DIM_DX_NORM         = 2;
constexpr int DIM_DY_NORM         = 3;
constexpr int DIM_HOPS_TAKEN      = 4;
constexpr int DIM_HOPS_REMAIN     = 5;
constexpr int DIM_HOPS_REMAIN_X   = 6;
constexpr int DIM_HOPS_REMAIN_Y   = 7;
constexpr int DIM_CREDIT_E        = 8;
constexpr int DIM_CREDIT_W        = 9;
constexpr int DIM_CREDIT_N        = 10;
constexpr int DIM_CREDIT_S        = 11;
constexpr int DIM_IS_BOUNDARY     = 12;
constexpr int DIM_INPORT_DIR      = 13;
// Self-congestion features (router's own state)
constexpr int DIM_SELF_VC_BUSY    = 14;  // % of input VCs with a flit ready
constexpr int DIM_SELF_CROSSBAR   = 15;  // % of inports with ready flits
constexpr int DIM_SELF_OUT_CREDITS = 16; // avg credit across all output VCs
constexpr int DIM_SELF_OUT_BUSY   = 17;  // % of output VCs non-IDLE

// Reward constants
constexpr float REWARD_BOUNDARY     = -1.0f;
constexpr float REWARD_SHORTEST     =  0.3f;
constexpr float REWARD_DETOUR       = -0.4f;
constexpr float REWARD_GLOBAL_BETA  =  0.3f;  // Weight for global latency penalty

/**
 * Build the 5-token × 14D state array for ADQN inference.
 *
 * @param router      The current router making the routing decision
 * @param route       RouteInfo with dest_router, hops_traversed, etc.
 * @param inport_dirn The direction the flit arrived from ("East","West",...)
 * @param tokens      Output: 5×14 float array filled with normalized features
 */
void buildTokenStates(Router *router, RouteInfo route,
                      PortDirection inport_dirn,
                      float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM]);

/**
 * Build the action mask (max 2 toward-destination directions).
 * mask[i] = 1 only if action i moves toward the destination AND
 *           that direction exists on the mesh.
 *
 * @param router      The current router
 * @param route       RouteInfo with destination router
 * @param action_mask Output: 4-element int array (E, W, N, S)
 */
void buildActionMask(Router *router, RouteInfo route,
                     int action_mask[ADQN_NUM_ACTIONS]);

/**
 * Compute immediate reward for the chosen action.
 * Since actions are restricted to toward-destination only,
 * rewards are always positive (progress toward dest + congestion bonus).
 *
 * @param action The chosen action (0=E, 1=W, 2=N, 3=S)
 * @param router The current router
 * @param route  RouteInfo with destination
 * @return Reward value
 */
float computeReward(int action, Router *router, RouteInfo route);

/**
 * Map an action (0..3) to the corresponding output port index.
 *
 * @param action The chosen action (0=E, 1=W, 2=N, 3=S)
 * @param router The current router (to access RoutingUnit's direction maps)
 * @return The output port index for that direction
 */
int actionToOutport(int action, Router *router);

/**
 * Get toward-destination neighbor router IDs for Q-value transfer.
 * Fills toward_ids[4] with router IDs (0=E, 1=W, 2=N, 3=S); -1 if none.
 */
void getTowardNeighborIds(Router *router, RouteInfo route,
                          int toward_ids[ADQN_NUM_ACTIONS]);

/**
 * Encode a PortDirection string to a normalized float [0, 1] for dim 13.
 * Encoding: East=0.0, West=0.25, North=0.5, South=0.75, Local=1.0
 */
float encodeInportDir(const PortDirection &dir);

/**
 * Get the reverse direction string (e.g., "East" → "West").
 */
PortDirection reverseDirection(const PortDirection &dir);

} // namespace garnet
} // namespace ruby
} // namespace gem5

#endif // __MEM_RUBY_NETWORK_GARNET_ADQN_ROUTING_HH__
