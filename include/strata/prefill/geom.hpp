// include/strata/prefill/geom.hpp - the prefill geometry, in ONE place.
//
// ADAPTED for Whittle-Next-27B-A3B (Flash-Next's values in brackets).  These numbers used to be literals
// scattered through prefill.cpp, kernels.cu and the moe_*.cu files - 130+ sites, where a missed one is a wrong
// buffer size or a wrong loop bound rather than a compile error.  One definition, included everywhere.
#pragma once

#include <cstdint>

namespace strata::prefill {
namespace geom {
inline constexpr int64_t NL = 40;                  // block_count                     (Flash-Next 48)
inline constexpr int64_t N = 2048;                 // n_embd                          (2560)
inline constexpr int64_t HC = 4;                   // hyper_connection.count
inline constexpr int64_t D = N * HC;               // hc_dim = 8192                   (10240)
inline constexpr int64_t LR = 320;                 // hyper_connection.low_rank
inline constexpr int64_t FF = 512;                 // expert_feed_forward_length      (640)
inline constexpr int64_t GU = 2 * FF;              // gate|up rows of one expert      (1280)
inline constexpr int64_t K = 8;                    // expert_used_count               (10)
inline constexpr int64_t NE = 180;                 // expert_count                    (512)
inline constexpr int64_t NH = 16;                  // attention.head_count            (24)
inline constexpr int64_t HD = 256;                 // attention.key_length = value_length
inline constexpr int64_t KVH = 2;                  // attention.head_count_kv
inline constexpr int64_t QD = NH * 2 * HD;         // q_full row width (q|gate) = 8192 (12288)
inline constexpr int64_t QW = NH * HD;             // q rows of one token = 4096       (6144)
inline constexpr int64_t KV = KVH * HD;            // K or V rows of one token = 512
inline constexpr int64_t S = 128;                  // ssm.state_size
inline constexpr int64_t HK = 16;                  // ssm.group_count
inline constexpr int64_t HV = 32;                  // ssm value heads                 (48)
inline constexpr int64_t C = 2 * S * HK + S * HV;  // conv channels = 8192            (10240)
inline constexpr int64_t ZV = S * HV;              // ssm value dim = 4096            (6144)
}  // namespace geom
}  // namespace strata::prefill