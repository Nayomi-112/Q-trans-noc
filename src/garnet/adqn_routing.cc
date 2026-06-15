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

#include "mem/ruby/network/garnet/adqn_routing.hh"

#include "base/compiler.hh"
#include <algorithm>
#include <cmath>

#include "mem/ruby/network/garnet/GarnetNetwork.hh"
#include "mem/ruby/network/garnet/InputUnit.hh"
#include "mem/ruby/network/garnet/OutputUnit.hh"
#include "mem/ruby/network/garnet/Router.hh"

namespace gem5
{

namespace ruby
{

namespace garnet
{

// Map action index to direction name
static PortDirection actionToDir(int action)
{
    switch (action) {
        case 0: return "East";
        case 1: return "West";
        case 2: return "North";
        case 3: return "South";
        default: return "Unknown";
    }
}

float
encodeInportDir(const PortDirection &dir)
{
    if (dir == "East")  return 0.0f;
    if (dir == "West")  return 0.25f;
    if (dir == "North") return 0.5f;
    if (dir == "South") return 0.75f;
    // "Local" or unknown → 1.0
    return 1.0f;
}

PortDirection
reverseDirection(const PortDirection &dir)
{
    if (dir == "East")  return "West";
    if (dir == "West")  return "East";
    if (dir == "North") return "South";
    if (dir == "South") return "North";
    return "Local";
}

// Find the output port index for a given direction on a router
static int
findOutportByDirection(Router *router, const PortDirection &dir)
{
    for (int i = 0; i < router->get_num_outports(); i++) {
        if (router->getOutportDirection(i) == dir)
            return i;
    }
    return -1;
}

// Sum credit counts across all VCs for a given output direction
static float
getDirectionCredits(Router *router, const PortDirection &dir)
{
    int outport = findOutportByDirection(router, dir);
    if (outport < 0)
        return 0.0f;

    OutputUnit *ou = router->getOutputUnit(outport);
    int num_vcs = router->get_num_vcs();
    int total_credits = 0;
    for (int vc = 0; vc < num_vcs; vc++) {
        total_credits += ou->get_credit_count(vc);
    }
    // Normalize by (num_vcs * max_credits_per_vc)
    int vc_per_vnet = router->get_vc_per_vnet();
    int num_vnets = router->get_num_vnets();
    int max_total = num_vnets * vc_per_vnet * (int)MAX_CREDITS;
    if (max_total == 0)
        return 0.0f;
    return (float)total_credits / (float)max_total;
}

// Compute self-congestion features for a router
static void
computeSelfCongestion(Router *router, float self_features[4])
{
    int num_inports = router->get_num_inports();
    int num_outports = router->get_num_outports();
    int num_vcs = router->get_num_vcs();
    int vc_per_vnet = router->get_vc_per_vnet();
    int num_vnets = router->get_num_vnets();
    Tick cur = curTick();

    // [0] DIM_SELF_VC_BUSY: fraction of input VCs with a flit ready
    int total_input_vcs = num_inports * num_vcs;
    int busy_input_vcs = 0;
    int inports_with_ready = 0;
    for (int p = 0; p < num_inports; p++) {
        InputUnit *iu = router->getInputUnit(p);
        bool port_has_ready = false;
        for (int vc = 0; vc < num_vcs; vc++) {
            if (iu->isReady(vc, cur)) {
                busy_input_vcs++;
                port_has_ready = true;
            }
        }
        if (port_has_ready)
            inports_with_ready++;
    }
    self_features[0] = (total_input_vcs > 0) ?
        (float)busy_input_vcs / (float)total_input_vcs : 0.0f;

    // [1] DIM_SELF_CROSSBAR: fraction of inports with at least 1 ready flit
    self_features[1] = (num_inports > 0) ?
        (float)inports_with_ready / (float)num_inports : 0.0f;

    // [2] DIM_SELF_OUT_CREDITS: average credit across all output VCs
    int total_credits = 0;
    int total_output_vcs = 0;
    for (int p = 0; p < num_outports; p++) {
        OutputUnit *ou = router->getOutputUnit(p);
        for (int vc = 0; vc < num_vcs; vc++) {
            total_credits += ou->get_credit_count(vc);
            total_output_vcs++;
        }
    }
    int max_total = num_vnets * vc_per_vnet * (int)MAX_CREDITS;
    self_features[2] = (max_total > 0 && total_output_vcs > 0) ?
        (float)total_credits / (float)(total_output_vcs * max_total) : 0.0f;

    // [3] DIM_SELF_OUT_BUSY: fraction of output VCs that are NOT idle
    int busy_output_vcs = 0;
    for (int p = 0; p < num_outports; p++) {
        OutputUnit *ou = router->getOutputUnit(p);
        for (int vc = 0; vc < num_vcs; vc++) {
            if (!ou->is_vc_idle(vc, cur))
                busy_output_vcs++;
        }
    }
    self_features[3] = (total_output_vcs > 0) ?
        (float)busy_output_vcs / (float)total_output_vcs : 0.0f;
}

// Build a single 18D token for a specific router
static void
buildSingleToken(Router *router, RouteInfo route,
                 PortDirection inport_dirn,
                 float token[ADQN_TOKEN_DIM])
{
    GarnetNetwork *net = router->get_net_ptr();
    int num_cols = net->getNumCols();
    int num_rows = net->getNumRows();

    int my_id = router->get_id();
    int my_x = (num_cols > 0) ? my_id % num_cols : 0;
    int my_y = (num_cols > 0) ? my_id / num_cols : 0;

    int dest_id = route.dest_router;
    int dest_x = (num_cols > 0) ? dest_id % num_cols : 0;
    int dest_y = (num_cols > 0) ? dest_id / num_cols : 0;

    int x_hops = abs(dest_x - my_x);
    int y_hops = abs(dest_y - my_y);
    int manhattan = x_hops + y_hops;
    int max_hops = num_rows + num_cols - 2;
    if (max_hops < 1) max_hops = 1;

    // [0] rx (raw, batch-normalized in Python)
    token[DIM_RX_NORM] = (float)my_x;
    // [1] ry (raw, batch-normalized in Python)
    token[DIM_RY_NORM] = (float)my_y;
    // [2] dx (raw, batch-normalized in Python)
    token[DIM_DX_NORM] = (float)dest_x;
    // [3] dy (raw, batch-normalized in Python)
    token[DIM_DY_NORM] = (float)dest_y;
    // [4] hops_taken_norm
    token[DIM_HOPS_TAKEN] = (float)route.hops_traversed / (float)max_hops;
    // [5] hops_remain_norm
    token[DIM_HOPS_REMAIN] = (float)manhattan / (float)max_hops;
    // [6] hops_remain_x (raw)
    token[DIM_HOPS_REMAIN_X] = (float)x_hops;
    // [7] hops_remain_y (raw)
    token[DIM_HOPS_REMAIN_Y] = (float)y_hops;
    // [8-11] credit_E/W/N/S
    token[DIM_CREDIT_E] = getDirectionCredits(router, "East");
    token[DIM_CREDIT_W] = getDirectionCredits(router, "West");
    token[DIM_CREDIT_N] = getDirectionCredits(router, "North");
    token[DIM_CREDIT_S] = getDirectionCredits(router, "South");
    // [12] is_boundary
    bool boundary = (my_x == 0 || my_x == num_cols - 1 ||
                     my_y == 0 || my_y == num_rows - 1);
    token[DIM_IS_BOUNDARY] = boundary ? 1.0f : 0.0f;
    // [13] inport_dir
    token[DIM_INPORT_DIR] = encodeInportDir(inport_dirn);

    // [14-17] self-congestion features
    float self_cong[4];
    computeSelfCongestion(router, self_cong);
    token[DIM_SELF_VC_BUSY]     = self_cong[0];
    token[DIM_SELF_CROSSBAR]    = self_cong[1];
    token[DIM_SELF_OUT_CREDITS] = self_cong[2];
    token[DIM_SELF_OUT_BUSY]    = self_cong[3];
}

void
buildTokenStates(Router *router, RouteInfo route,
                 PortDirection inport_dirn,
                 float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM])
{
    // Build 3 tokens: current + 2 toward-destination neighbors
    GarnetNetwork *net = router->get_net_ptr();
    int num_cols = net->getNumCols();
    int num_rows = net->getNumRows();
    int my_id = router->get_id();
    int my_x = (num_cols > 0) ? my_id % num_cols : 0;
    int my_y = (num_cols > 0) ? my_id / num_cols : 0;

    buildSingleToken(router, route, inport_dirn, tokens[TOKEN_CURRENT]);

    int dest_id = route.dest_router;
    int dest_x = (num_cols > 0) ? dest_id % num_cols : 0;
    int dest_y = (num_cols > 0) ? dest_id / num_cols : 0;
    int dx_diff = dest_x - my_x;
    int dy_diff = dest_y - my_y;

    // Zero-fill neighbor tokens
    for (int t = 1; t < ADQN_NUM_TOKENS; t++)
        for (int d = 0; d < ADQN_TOKEN_DIM; d++)
            tokens[t][d] = 0.0f;

    int token_idx = 1;
    if (dx_diff > 0 && my_x + 1 < num_cols) {
        Router *neighbor = net->getRouter(my_id + 1);
        buildSingleToken(neighbor, route, "West", tokens[token_idx++]);
    } else if (dx_diff < 0 && my_x - 1 >= 0) {
        Router *neighbor = net->getRouter(my_id - 1);
        buildSingleToken(neighbor, route, "East", tokens[token_idx++]);
    }
    if (dy_diff > 0 && my_y + 1 < num_rows) {
        Router *neighbor = net->getRouter(my_id + num_cols);
        buildSingleToken(neighbor, route, "South", tokens[token_idx++]);
    } else if (dy_diff < 0 && my_y - 1 >= 0) {
        Router *neighbor = net->getRouter(my_id - num_cols);
        buildSingleToken(neighbor, route, "North", tokens[token_idx++]);
    }
}

void
buildActionMask(Router *router, RouteInfo route,
                int action_mask[ADQN_NUM_ACTIONS])
{
    GarnetNetwork *net = router->get_net_ptr();
    int num_cols = net->getNumCols();
    int num_rows = net->getNumRows();
    int my_id = router->get_id();
    int my_x = (num_cols > 0) ? my_id % num_cols : 0;
    int my_y = (num_cols > 0) ? my_id / num_cols : 0;

    int dest_id = route.dest_router;
    int dest_x = (num_cols > 0) ? dest_id % num_cols : 0;
    int dest_y = (num_cols > 0) ? dest_id / num_cols : 0;

    int dx_diff = dest_x - my_x;
    int dy_diff = dest_y - my_y;

    // Toward directions (boundary-safe)
    bool toward_e = (my_x < num_cols - 1 && dx_diff > 0);
    bool toward_w = (my_x > 0 && dx_diff < 0);
    bool toward_n = (my_y < num_rows - 1 && dy_diff > 0);
    bool toward_s = (my_y > 0 && dy_diff < 0);

    // Check toward-direction credits
    float credit_e = toward_e ? getDirectionCredits(router, "East")  : 1.0f;
    float credit_w = toward_w ? getDirectionCredits(router, "West")  : 1.0f;
    float credit_n = toward_n ? getDirectionCredits(router, "North") : 1.0f;
    float credit_s = toward_s ? getDirectionCredits(router, "South") : 1.0f;

    // Threshold: allow detours if ALL toward directions are congested
    constexpr float CREDIT_THRESHOLD = 0.25f;
    bool any_toward_free = (toward_e && credit_e >= CREDIT_THRESHOLD)
                        || (toward_w && credit_w >= CREDIT_THRESHOLD)
                        || (toward_n && credit_n >= CREDIT_THRESHOLD)
                        || (toward_s && credit_s >= CREDIT_THRESHOLD);

    if (any_toward_free) {
        // Normal mode: only toward directions
        action_mask[0] = toward_e ? 1 : 0;
        action_mask[1] = toward_w ? 1 : 0;
        action_mask[2] = toward_n ? 1 : 0;
        action_mask[3] = toward_s ? 1 : 0;
    } else {
        // Congestion escape: allow all boundary-valid directions
        action_mask[0] = (my_x < num_cols - 1) ? 1 : 0; // East
        action_mask[1] = (my_x > 0) ? 1 : 0;            // West
        action_mask[2] = (my_y < num_rows - 1) ? 1 : 0; // North
        action_mask[3] = (my_y > 0) ? 1 : 0;            // South
    }
}

static PortDirection
shortestDirToward(int my_x, int my_y, int dest_x, int dest_y)
{
    int dx = dest_x - my_x;
    int dy = dest_y - my_y;
    if (abs(dx) >= abs(dy))
        return (dx > 0) ? "East" : "West";
    else
        return (dy > 0) ? "North" : "South";
}

float
computeReward(int action, Router *router, RouteInfo route)
{
    GarnetNetwork *net = router->get_net_ptr();
    int num_cols = net->getNumCols();
    int num_rows = net->getNumRows();
    int my_id = router->get_id();
    int my_x = (num_cols > 0) ? my_id % num_cols : 0;
    int my_y = (num_cols > 0) ? my_id / num_cols : 0;

    int dest_id = route.dest_router;
    int dest_x = (num_cols > 0) ? dest_id % num_cols : 0;
    int dest_y = (num_cols > 0) ? dest_id / num_cols : 0;

    int old_dist = abs(dest_x - my_x) + abs(dest_y - my_y);

    // Boundary check (safety; action mask should prevent this)
    switch (action) {
        case 0: if (my_x >= num_cols - 1) return REWARD_BOUNDARY; break;
        case 1: if (my_x <= 0)            return REWARD_BOUNDARY; break;
        case 2: if (my_y >= num_rows - 1) return REWARD_BOUNDARY; break;
        case 3: if (my_y <= 0)            return REWARD_BOUNDARY; break;
    }

    // Compute new Manhattan distance (always toward destination)
    int new_x = my_x, new_y = my_y;
    switch (action) {
        case 0: new_x++; break;
        case 1: new_x--; break;
        case 2: new_y++; break;
        case 3: new_y--; break;
    }
    int new_dist = abs(dest_x - new_x) + abs(dest_y - new_y);

    // Credit on the chosen direction (0=congested, 1=free)
    PortDirection chosen_dir = actionToDir(action);
    float chosen_credit = getDirectionCredits(router, chosen_dir);

    // Credit-scaled progress: strongly incentivize choosing free directions
    float progress = (float)(old_dist - new_dist) / (float)old_dist;
    float credit_factor = 0.2f + 0.8f * chosen_credit;  // ranges 0.2 (congested) to 1.0 (free)
    float base_reward = (0.05f + 0.15f * progress) * credit_factor;

    // Local output congestion penalty: discourage choosing busy output ports
    float self_cong[4];
    computeSelfCongestion(router, self_cong);
    float out_busy = self_cong[3];  // DIM_SELF_OUT_BUSY
    float local_penalty = 0.15f * out_busy;

    // Global latency penalty (reduced weight to let local signals dominate)
    float global_ema = net->getGlobalLatencyEMA();
    int max_hops = num_rows + num_cols - 2;
    float global_penalty = 0.15f * (global_ema / (float)max_hops);

    return base_reward - local_penalty - global_penalty;
}

int
actionToOutport(int action, Router *router)
{
    PortDirection dir = actionToDir(action);
    for (int i = 0; i < router->get_num_outports(); i++) {
        if (router->getOutportDirection(i) == dir)
            return i;
    }
    // Fallback: should not happen if topology is a proper mesh
    return 0;
}

void
getTowardNeighborIds(Router *router, RouteInfo route,
                     int toward_ids[ADQN_NUM_ACTIONS])
{
    GarnetNetwork *net = router->get_net_ptr();
    int num_cols = net->getNumCols();
    int my_id = router->get_id();
    int my_x = (num_cols > 0) ? my_id % num_cols : 0;
    int my_y = (num_cols > 0) ? my_id / num_cols : 0;

    int dest_id = route.dest_router;
    int dest_x = (num_cols > 0) ? dest_id % num_cols : 0;
    int dest_y = (num_cols > 0) ? dest_id / num_cols : 0;

    int dx_diff = dest_x - my_x;
    int dy_diff = dest_y - my_y;

    toward_ids[0] = (my_x < num_cols - 1 && dx_diff > 0) ? my_id + 1 : -1;
    toward_ids[1] = (my_x > 0 && dx_diff < 0)            ? my_id - 1 : -1;
    toward_ids[2] = (my_y < (num_cols * (net->getNumRows())) / net->getNumRows() && dy_diff > 0)
                    ? -1 : -1;  // fallback
    // Simpler: direct neighbor in mesh
    int num_rows = net->getNumRows();
    toward_ids[0] = (my_x < num_cols - 1 && dx_diff > 0) ? my_id + 1 : -1;
    toward_ids[1] = (my_x > 0 && dx_diff < 0)            ? my_id - 1 : -1;
    toward_ids[2] = (my_y < num_rows - 1 && dy_diff > 0) ? my_id + num_cols : -1;
    toward_ids[3] = (my_y > 0 && dy_diff < 0)            ? my_id - num_cols : -1;
}

} // namespace garnet
} // namespace ruby
} // namespace gem5
