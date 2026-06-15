#include "mem/ruby/network/garnet/zmq_functions.hh"
#ifdef HAVE_ZMQ
#include <zmq.h>
#include "mem/ruby/network/garnet/GarnetNetwork.hh"
#include <cassert>
#include <cstring>
#include <string>
#include <cmath>
namespace gem5 { namespace ruby { namespace garnet {

static void *g_zmq_context = nullptr;
static void *g_zmq_dealer = nullptr;
static void *g_zmq_push_socket = nullptr;
static bool g_zmq_initialized = false;

void zmq_init(int decision_port, int experience_port)
{
    if (g_zmq_initialized) return;
    g_zmq_context = zmq_ctx_new(); assert(g_zmq_context);
    g_zmq_dealer = zmq_socket(g_zmq_context, ZMQ_REQ); assert(g_zmq_dealer);
    int timeout_ms = 5000;
    zmq_setsockopt(g_zmq_dealer, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    std::string addr = "tcp://localhost:" + std::to_string(decision_port);
    int rc = zmq_connect(g_zmq_dealer, addr.c_str()); assert(rc == 0);
    g_zmq_push_socket = zmq_socket(g_zmq_context, ZMQ_PUSH); assert(g_zmq_push_socket);
    std::string exp_addr = "tcp://localhost:" + std::to_string(experience_port);
    rc = zmq_connect(g_zmq_push_socket, exp_addr.c_str()); assert(rc == 0);
    g_zmq_initialized = true;
}

// #define ADQN_NATIVE_INFERENCE
// #define ADQN_Q_TRANSFER

#ifdef ADQN_NATIVE_INFERENCE
#include "/home/nayomi/gem5/adqn_results/mlp_weights.h"

int zmq_send_inference(const float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
                       const int action_mask[ADQN_NUM_ACTIONS],
                       int router_id, const int toward_ids[ADQN_NUM_ACTIONS], GarnetNetwork *net)
{
    constexpr int IN = 54, H1 = 32, H2 = 16, OUT = 4;
    float input[IN]; int idx = 0;
    for (int t = 0; t < ADQN_NUM_TOKENS; t++)
        for (int d = 0; d < ADQN_TOKEN_DIM; d++)
            input[idx++] = (tokens[t][d] - bn_running_mean[d])
                          / std::sqrt(bn_running_var[d] + 1e-5f)
                          * bn_weight[d] + bn_bias[d];
    float h1[H1];
    for (int i = 0; i < H1; i++) {
        float s = fc1_bias[i];
        for (int j = 0; j < IN; j++) s += fc1_weight[i*IN+j] * input[j];
        h1[i] = (s > 0.0f) ? s : 0.0f;
    }
    float h2[H2];
    for (int i = 0; i < H2; i++) {
        float s = fc2_bias[i];
        for (int j = 0; j < H1; j++) s += fc2_weight[i*H1+j] * h1[j];
        h2[i] = (s > 0.0f) ? s : 0.0f;
    }
    float q[OUT];
    for (int i = 0; i < OUT; i++) {
        float s = fc3_bias[i];
        for (int j = 0; j < H2; j++) s += fc3_weight[i*H2+j] * h2[j];
        q[i] = s;
    }
    // Q-transfer (enable via #define ADQN_Q_TRANSFER)
#ifdef ADQN_Q_TRANSFER
    constexpr float LAMBDA = 0.04f;
    // Store q_local BEFORE modification (matches Python StateBuffer)
    net->setRouterPrevQ(router_id, q);
    // Read neighbors' previous q_local and augment
    float q_down[OUT] = {0,0,0,0};
    for (int d = 0; d < OUT; d++) {
        if (toward_ids[d] >= 0) {
            float prev_q[4];
            net->getRouterPrevQ(toward_ids[d], prev_q);
            q_down[d] = prev_q[d];
        }
    }
    for (int i = 0; i < OUT; i++) q[i] += LAMBDA * q_down[i];
#endif
    int action = 0; float best = -1e30f;
    for (int i = 0; i < OUT; i++)
        if (action_mask[i] > 0 && q[i] > best) { best = q[i]; action = i; }
    return action;
}

#else
int zmq_send_inference(const float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
                       const int action_mask[ADQN_NUM_ACTIONS],
                       int router_id, const int toward_ids[ADQN_NUM_ACTIONS], GarnetNetwork *net)
{
    constexpr int fs = ADQN_NUM_TOKENS * ADQN_TOKEN_DIM * sizeof(float);
    constexpr int ms = ADQN_NUM_ACTIONS * sizeof(int), mgs = fs + ms + sizeof(int);
    char msg[mgs];
    std::memcpy(msg, tokens, fs);
    std::memcpy(msg + fs, action_mask, ms);
    std::memcpy(msg + fs + ms, &router_id, sizeof(int));
    zmq_send(g_zmq_dealer, msg, mgs, 0);
    char reply[sizeof(int)];
    int rc = zmq_recv(g_zmq_dealer, reply, sizeof(int), 0);
    if (rc <= 0) return 0;
    int action; std::memcpy(&action, reply, sizeof(int)); return action;
}
#endif

int zmq_send_inference_batch(int router_id,
    const float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
    const int action_mask[ADQN_NUM_ACTIONS], float&, bool&) {
    int dummy[4]={-1,-1,-1,-1};
    return zmq_send_inference(tokens, action_mask, router_id, dummy, nullptr);
}
void zmq_flush_batch() {}

void zmq_send_experience(const float tokens[ADQN_NUM_TOKENS][ADQN_TOKEN_DIM],
                         int action, float reward, bool terminal)
{
    constexpr int fs = ADQN_NUM_TOKENS * ADQN_TOKEN_DIM * sizeof(float);
    constexpr int ms = fs + sizeof(int) + sizeof(float) + 1;
    char msg[ms];
    std::memcpy(msg, tokens, fs);
    std::memcpy(msg + fs, &action, sizeof(int));
    std::memcpy(msg + fs + sizeof(int), &reward, sizeof(float));
    msg[fs + sizeof(int) + sizeof(float)] = terminal ? 1 : 0;
    zmq_send(g_zmq_push_socket, msg, ms, ZMQ_DONTWAIT);
}

void zmq_cleanup() {
    if (g_zmq_dealer) { zmq_close(g_zmq_dealer); g_zmq_dealer = nullptr; }
    if (g_zmq_push_socket) { zmq_close(g_zmq_push_socket); g_zmq_push_socket = nullptr; }
    if (g_zmq_context) { zmq_ctx_destroy(g_zmq_context); g_zmq_context = nullptr; }
    g_zmq_initialized = false;
}

}}} // namespace gem5::ruby::garnet
#endif // HAVE_ZMQ
