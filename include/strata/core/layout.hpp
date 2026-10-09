// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// The model's geometry.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
///
/// ADAPTED 2026-10-07 for `logic65/Whittle-Next-27B-A3B` (general.architecture `qwen4exp`,
/// general.size_label `180x5.0B`, v4.4 Q4_K_M).  NOTHING BELOW IS COPIED FROM `docs/semantics.md` - that
/// file describes Flash-Next, the model this engine shipped with.  Each value here is read from the
/// Whittle artifact's own GGUF metadata, and the fields that did NOT change are left untouched so that a
/// diff of this struct is exactly the geometry delta between the two models.
///
///   block_count 40        embedding_length 2048      expert_count 180     expert_used_count 8
///   expert_feed_forward_length 512   attention.head_count 16   head_count_kv 2   key/value_length 256
///   ssm.{state_size 128, group_count 16, time_step_rank 32, inner_size 4096, conv_kernel 4}
///   hyper_connection.{count 4, low_rank 320}   attention.indexer.{head_count 4, key_length 128}
///
/// Cross-checked against tensor shapes in that same file: `blk.0.ssm_conv1d.weight` = [4, 8192],
/// `blk.1.ple_key.weight` = [2048, 8192], `output_hc_up.weight` = [320, 8192].
struct ModelGeometry {
    int64_t n_embd = 2048;
    int64_t n_layers = 40;
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 39

    // GDN (30 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 32;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 8192;    ///< 2*128*16 + 128*32
    int64_t ssm_value_dim = 4096;        ///< 128 * 32

    // QSA (10 layers)
    int64_t n_head = 16;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;
    /// `attention.indexer.top_k`: the selection budget in cells.  Whittle-Next: 262144 (= 2^18, the whole
    /// context, so nothing is ever pruned); Flash-Next: 2048.
    int64_t idx_top_k = 262144;
    /// `attention.compress_ratios` for a full-attention layer: the indexer pools ONE key per this many cells.
    /// **0 MEANS DENSE, AND DENSE IS NOT A DETAIL** - the reference gates its entire QSA path on
    /// `compress_ratios[il] > 0` (`third_party/llama.cpp/src/models/qwen4exp.cpp:786`), so a model whose
    /// ratios are all zero has no block compression and no indexer sparsity at all.  Whittle-Next: 0 on all
    /// 40 layers.  Flash-Next: 4 on its 12 QSA layers - which is exactly the `idx_block = 4` that
    /// `kernels::qsa_real_shapes()` carries.
    int64_t qsa_compress_ratio = 0;

    // gated residual, on every layer
    // ADAPTED 2026-10-08 for Cyber-Tiel-Coder-35B-A3B: **`hc = 1` IS THE PLAIN-RESIDUAL DECLARATION.**
    // Cyber (like every ordinary transformer) has NO hyper-connections: one residual stream, `attn_norm`
    // before attention and `post_attention_norm` before the FFN.  Strata's shipped geometry is the
    // 4-stream GATED residual (`hc = 4`, `hc_lr = 320`, six `hc_*` projections per layer), which Cyber's
    // file does not carry at all - `src/core/layer.cpp:1231` would refuse the load with
    // "blk.0.hc_attn_{norm,down,up,inject}.weight is missing" (41 layers x 8 tensors = 328).
    //
    // `hc == 1` is the switch: every site that would read a `hc_*` projection instead treats the two
    // per-layer norms as ordinary weighted RMSNorms on ONE stream and the residual write as a plain add.
    // This is the same transformation `src/core/mtp.cpp` already performs for a donor MTP block under
    // `STRATA_MTP_PLAIN=1` (see its `:108-116` note and the `if (mtp_plain())` branches at `:585`/`:648`/
    // `:703`) - copied rather than reinvented, so the two paths stay recognisably the same.
    // `hc_lr` is left at 320: it now only sizes the (unused in plain mode) GR workspace.
    int64_t hc = 1;
    int64_t hc_lr = 320;

    /// True when the target has no hyper-connections: one residual stream, standard pre/post-norm.
    /// Every plain-mode branch keys off this rather than off a separate flag, so there is exactly one
    /// place that decides - and `ModelGeometry` is in scope at every site that needs it.
    bool plain_residual() const { return hc == 1; }

    // MoE, on every layer
    // ADAPTED 2026-10-08 for Cyber-Tiel-Coder-35B-A3B (`qwen35moe`): `expert_count` 256 and
    // `expert_used_count` 8, both read straight out of the GGUF.  The file says `block_count` 41, but that
    // counts `blk.40` - the MTP block - so **n_layers stays 40**: the MTP block is a separate draft
    // artifact (`core::MtpDrafter` + `draft_geometry`), never part of this geometry.  That is also the
    // reference implementation's convention: `qwen35moe.cpp:28-33` calls `n_layer() == 40` "35B_A3B" and
    // keeps `n_layer_all` (which includes MTP) for the recurrent-layer array alone.
    int64_t n_expert = 256;
    int64_t n_ff = 512;

    int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
