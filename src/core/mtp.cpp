// src/core/mtp.cpp - see include/strata/core/mtp.hpp.
#include "strata/core/mtp.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/on_device.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <vector>
#if defined(_WIN32)
#include <intrin.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

// 64-bit seek/tell on a `FILE*`: `fseek`/`ftell` take a 32-bit `long` on Windows and would wrap past 2 GiB.
#if defined(_WIN32)
#define STRATA_FILE_SEEK64(f, o, w) _fseeki64((f), (long long) (o), (w))
#define STRATA_FILE_TELL64(f) _ftelli64(f)
#else
#define STRATA_FILE_SEEK64(f, o, w) fseeko((f), (off_t) (o), (w))
#define STRATA_FILE_TELL64(f) ftello(f)
#endif

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    // Loader fix (0.1.15+loaderfix.2): `ifstream::read` reaches the disk as 4095-byte reads on MSVC - the same
    // split `load_experts_ranges` had - so the drafter's 111 MiB dense blob paid 4 KiB per operation on a cold
    // start.  `fread` passes a request bigger than the stream buffer straight to `ReadFile()`.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    // A `FILE*` has no destructor and the short-read path below returns early: the guard closes on every path.
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
    if (STRATA_FILE_SEEK64(f, 0, SEEK_END) != 0) return false;
    const long long n = (long long) STRATA_FILE_TELL64(f);
    if (n < 0 || STRATA_FILE_SEEK64(f, 0, SEEK_SET) != 0) return false;
    out.resize((size_t) n);
    return n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n;
}

// ADAPTED: STRATA_MTP_PLAIN=1 runs a GRAFTED PLAIN draft block.  A donor trained on one stream (the
// Qwen3.6-generation MTP head: mtp.fc + input_layernorm / post_attention_layernorm / norm) has no
// hyper-connections at all, so every hc read and write below becomes a plain RMSNorm (or a plain residual
// add) on a single stream, and the body's multi-stream residual is collapsed by summing its streams -
// RMSNorm is scale-invariant, so the sum normalises exactly as the mean would.  Unset = unchanged.
// ADAPTED (2026-10-09, THIS PORT): the staging buffer for the per-stage snapshots taken inside the recorded
// graph (STRATA_MTP_DUMP_STAGES).  Device memory; read out by the debug block, never in a captured region.
float* g_stage_dev = nullptr;
// ADAPTED (2026-10-10, THIS PORT): the 0th hop's head-logit row, snapshotted INSIDE the graph.  `head_logits_` is
// recomputed by every full pass (the round's own row and then one per chain step), so the read-out in `draft()` was
// always looking at the LAST hop's distribution while the staged seed/head-input slots belong to the 0th hop - a
// same-hop comparison is only possible if the 0th hop's row is copied aside where later hops cannot overwrite it.
float* g_head_dev = nullptr;

static bool mtp_plain() {
    static const bool v = [] { const char* e = std::getenv("STRATA_MTP_PLAIN"); return e != nullptr && e[0] == '1'; }();
    return v;
}

}  // namespace

MtpDrafter::~MtpDrafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : prefill_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : prefill_dev_exec_) if (e) cudaGraphExecDestroy(e);
    if (pf_dev_) cudaFree(pf_dev_);
    for (auto& e : round_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : round_exec_c_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_c_) if (e) cudaGraphExecDestroy(e);
    if (cparams_) cudaFree(cparams_);
    if (cring_) cudaFree(cring_);
    if (dinv_) cudaFree(dinv_);
    if (cscratch_) cudaFree(cscratch_);
    if (h_cparams_) cudaFreeHost(h_cparams_);
    if (h_chist_) cudaFreeHost(h_chist_);
    if (cs_) cudaStreamDestroy(cs_);
    if (dense_) cudaFree(dense_);
    if (experts_) cudaFree(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (dhead_) cudaFree(dhead_);
    if (dvocab_) cudaFree(dvocab_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}

bool MtpDrafter::load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
                      int64_t window) {
    cudaGetDevice(&device_);   // a layer split's last stage on another GPU: the drafter lives there
    g_ = &g;
    ss_ = &ss;
    if (ple_ss_ == nullptr) ple_ss_ = &ss;
    max_t_ = max_t;
    rt_dir_ = rt_dir;
    if (max_t < 1 || max_t > strata::kernels::kVerifyMaxT) { err = "mtp: max_t out of range"; return false; }
    // Loader fix (0.1.15+loaderfix.2): the two reads below are the whole 闁炽儲绔竢after files闁?cost; reporting
    // them apart from the rest of the stage is what makes the next regression visible.
    const auto t_files = std::chrono::steady_clock::now();
    // ---- the index and the dense weights
    {
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
        std::string line;
        ne_ = (int) g.n_expert;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            // ADAPTED: `experts N` lets a grafted draft layer keep its trained routed-expert count, which
            // need not equal the model geometry's.  A draft layer copied verbatim from a sibling model
            // ships the donor's own router and experts; forcing them into the target's count would mean
            // cutting trained experts and rewriting the router, i.e. not verbatim at all.
            {
                std::istringstream probe(line);
                std::string k;
                probe >> k;
                if (k == "experts") {
                    int n = 0;
                    probe >> n;
                    if (n > 0) ne_ = n;
                    continue;
                }
            }
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
        const cudaError_t alloc = cudaMalloc((void**) &dense_, blob.size());
        if (alloc != cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            const cudaError_t info = cudaMemGetInfo(&free_bytes, &total_bytes);
            err = "mtp: dense weights allocation failed (" + std::string(cudaGetErrorString(alloc)) +
                  "), requested " + std::to_string(blob.size() >> 20) + " MiB, CUDA0 free " +
                  (info == cudaSuccess ? std::to_string(free_bytes >> 20) + " MiB" : "unknown");
            return false;
        }
        cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        vram_ += blob.size();
    }
    // ---- the 512 routed experts, one blob each
    {
        const uint64_t bytes = (uint64_t) ne_ * strata::kernels::cpu::BLOB;
        // Loader fix (0.1.15+loaderfix.2): each 64 MiB read below reached the disk as ~16k 4095-byte reads under
        // MSVC's `basic_filebuf::xsgetn`, which is what made 675 MiB of drafter experts take minutes.
        FILE* f = std::fopen((rt_dir + "/experts.bin").c_str(), "rb");
        if (f == nullptr) { err = "mtp: cannot open experts.bin"; return false; }
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        if (cudaMalloc((void**) &experts_, bytes) != cudaSuccess) { err = "mtp: the 512 experts do not fit in VRAM"; return false; }
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (std::fread(chunk.data(), 1, (size_t) n, f) != (size_t) n) { err = "mtp: experts.bin is truncated"; return false; }
            cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice);
            off += n;
        }
        vram_ += bytes;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }

    // ---- geometry self-check.  The table above carries the DONOR's own shapes while the forward below
    // indexes them with THIS body's geometry.  A donor that does not match used to load and run silently:
    // wrong bytes, no error, acceptance 0 - an unreadable zero.  Seen twice (闂傚啳鍩栭?1's 12 variants and the
    // 2560-wide Flash-Next head on 2026-10-06).  Refuse loudly instead, with name, expected and got.
    {
        const int64_t N = g.n_embd, HC = g.hc, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
        const bool plain = mtp_plain();
        struct Want { const char* name; int64_t rows, cols; bool norm; };
        const Want want[] = {
            {"fc_embedding.weight", N, N, false},
            {"fc_hidden.weight", N, N, false},
            {"pre_fc_norm_embedding.weight", 1, N, false},
            {"pre_fc_norm_hidden.weight", 1, plain ? N : HC * N, plain},
            {"attn_hyper_connection.hc_norm.weight", 1, plain ? N : HC * N, plain},
            {"mlp_hyper_connection.hc_norm.weight", 1, plain ? N : HC * N, plain},
            {"hyper_connection_mixer.hc_norm.weight", 1, plain ? N : HC * N, plain},
            {"self_attn.q_proj.weight", NH * 2 * HD, N, false},
            {"self_attn.o_proj.weight", N, NH * HD, false},
            {"self_attn.k_proj.weight", NKV * HD, N, false},
            {"self_attn.v_proj.weight", NKV * HD, N, false},
            {"mlp.gate.weight", (int64_t) ne_, N, false},
        };
        for (const Want& w : want) {
            for (const Tensor& t : tensors_) {
                if (t.name != w.name) continue;
                // a plain run normalises one stream: an N-wide norm and the packing tool's replicated
                // HC*N norm are both fine (it reads the first N).
                const bool replicated = w.norm && plain && t.rows == 1 && t.cols == HC * N;
                if (t.rows != w.rows || (t.cols != w.cols && !replicated)) {
                    err = "mtp: draft tensor " + t.name + " is " + std::to_string(t.rows) + "x" + std::to_string(t.cols) +
                          ", this body needs " + std::to_string(w.rows) + "x" + std::to_string(w.cols) +
                          (plain ? " (STRATA_MTP_PLAIN=1)" : "") + " - the donor block does not match the geometry";
                    return false;
                }
                break;
            }
        }
        if (plain) {   // self-evidence: the run must be able to show which path it took
            std::fprintf(stderr, "strata mtp: PLAIN donor path (STRATA_MTP_PLAIN=1): one stream, N=%lld hc=%lld experts=%d\n",
                         (long long) N, (long long) HC, ne_);
        }
    }

    // ---- the layer's own K/V (dense attention: no indexer state is read)
    const strata::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    // KV streaming: the drafter only reads its last `window` cells, so with streaming on its K/V is a ring of the
    // window (plus the cells a round writes ahead of its queries) over a host copy, refilled on a resume. The host copy
    // is pinned after the expert arena has pinned what it could: if it does not fit, the K/V stays whole in VRAM.
    int64_t ring = (window > 0 && window < max_cells) ? window + 4 * (int64_t) max_t + 64 : 0;
    // K8V4 never applies to the drafter: its own attention paths (below, and verify.cpp) handle whole formats
    // only, whatever ring shape it takes (0, a window, or the -1 fully-resident fallback).
    const bool kv_hybrid_was = qsa_kv_hybrid();
    const bool kv_int8_was = qsa_kv_int8();
    qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) qsa_set_kv_int8(true);   // the drafter under --kv k8v4: plain INT8
    uint64_t sb = qsa_state_bytes(g, max_cells, false, ring);
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
        if (st_.kv_mode == 0) { err = "mtp: state init failed"; return false; }
        std::fprintf(stderr, "strata mtp: no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM\n");
        cudaGetLastError();
        cudaFree(state_arena_);
        st_ = QsaState{};
        ring = -1;   // fully resident
        sb = qsa_state_bytes(g, max_cells, false, ring);
        if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
        if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) { err = "mtp: state init failed"; return false; }
    }
    qsa_set_kv_int8(kv_int8_was);
    qsa_set_kv_hybrid(kv_hybrid_was);
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_);
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) ne_); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
        sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        dummy_inj_ = b.take<float>(HC);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) { err = "mtp: stream"; return false; }
    const double files_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_files).count();
    std::fprintf(stderr, "strata mtp: draft layer loaded, %.0f MiB of VRAM (experts %.0f, dense %.0f), files read in %.2f s (%.0f MiB/s)\n",
                 (double) vram_ / 1048576.0, (double) ne_ * strata::kernels::cpu::BLOB / 1048576.0,
                 (double) tensors_.back().off / 1048576.0, files_s,
                 files_s > 0 ? ((double) ne_ * strata::kernels::cpu::BLOB + (double) tensors_.back().off) /
                                   1048576.0 / files_s : 0.0);
    return true;
}

uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const {
    uint64_t bytes = head_logits_ ? 0 : (uint64_t) max_t_ * (uint64_t) n_vocab * sizeof(float);
    if (dhead_ == nullptr) {
        if (FILE* f = std::fopen((rt_dir_ + "/draft_vocab.bin").c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long size = std::ftell(f);
            std::fclose(f);
            if (size >= 4 && size % 4 == 0) bytes += (uint64_t) (size / 4) * head_row_bytes + (uint64_t) size;
        }
    }
    // coupled draft sampling (STRATA_SPEC_COUPLED=1 only): an upper bound - the id -> subset map, the penalty ring,
    // the split scratch (64 lists of 64 entries at most) and the parameters
    if (coupled_draft_env() && cparams_ == nullptr)
        bytes += (uint64_t) n_vocab * 4 + (uint64_t) (kCoupledHistCap + max_t_) * 4 + 64ull * 64ull * 8ull + 4096;
    return bytes;
}

bool MtpDrafter::setup_coupled(std::string& err) {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    const size_t scratch = strata::kernels::coupled_draft_scratch_bytes((int) nv);
    if (scratch == 0) {
        std::fprintf(stderr, "strata mtp: STRATA_SPEC_COUPLED: %lld draft logits are too wide for the coupled sampler; "
                             "argmax drafts\n", (long long) nv);
        return true;
    }
    const size_t ring = (size_t) (kCoupledHistCap + max_t_) * sizeof(int32_t);
    if (cudaMalloc((void**) &cparams_, sizeof(strata::kernels::SamplerParams)) != cudaSuccess ||
        cudaMalloc((void**) &cring_, ring) != cudaSuccess || cudaMalloc(&cscratch_, scratch) != cudaSuccess ||
        !mapped(sizeof(strata::kernels::SamplerParams), (void**) &h_cparams_, (void**) &m_cparams_) ||
        !mapped((size_t) kCoupledHistCap * sizeof(int32_t), (void**) &h_chist_, (void**) &m_chist_)) {
        err = "mtp: the coupled draft sampler's buffers do not fit";
        return false;
    }
    cudaMemset(cring_, 0xff, ring);   // -1: no token
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
    vram_ += sizeof(strata::kernels::SamplerParams) + ring + scratch;
    if (dhead_ != nullptr) {   // token id -> subset index (-1: not in the draft head), for the penalties
        std::vector<int32_t> sub((size_t) n_dvocab_), inv((size_t) n_vocab_, -1);
        cudaMemcpy(sub.data(), dvocab_, sub.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < sub.size(); ++i)
            if (sub[i] >= 0 && sub[i] < n_vocab_ && inv[(size_t) sub[i]] < 0) inv[(size_t) sub[i]] = (int32_t) i;
        if (cudaMalloc((void**) &dinv_, inv.size() * sizeof(int32_t)) != cudaSuccess) {
            err = "mtp: the coupled draft sampler's token map does not fit";
            return false;
        }
        cudaMemcpy(dinv_, inv.data(), inv.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        vram_ += inv.size() * sizeof(int32_t);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("mtp: coupled setup: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    coupled_ok_ = true;
    std::fprintf(stderr, "strata mtp: coupled draft sampling on (STRATA_SPEC_COUPLED): sampled requests draft with the "
                         "target's chain and Philox draw over %lld tokens\n", (long long) nv);
    return true;
}

void MtpDrafter::set_draft_sampling(const strata::kernels::SamplerParams& sp) {
    coupled_active_ = coupled_ok_ && !sp.greedy && sp.temperature > 0.0f;
    if (!coupled_active_) return;
    *h_cparams_ = sp;   // read by the next round graph (after the previous one has synced)
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
}

void MtpDrafter::set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next) {
    if (!coupled_active_) return;
    const int h = coupled_hist_len(h_cparams_->penalty_last_n, kCoupledHistCap);
    coupled_hist_base(tail, n_tail, next, h, h_chist_ + (kCoupledHistCap - h));
}

namespace {
// #474: what to change when the draft head's token subset does not fit the VRAM left - a smaller subset that setup
// ships (data/draft_vocab_*.bin; their token counts below), and how to pick it.  The start used to stop at "the draft
// head does not fit" with no hint.  Text only, after the failure: nothing changes for a start that fits.
void draft_head_hint(int64_t n_tokens, int64_t row_bytes) {
    size_t free_b = 0, total_b = 0;
    const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    auto mib = [&](int64_t n) { return (double) (n * row_bytes) / 1048576.0; };
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    std::fprintf(stderr, "strata mtp: the draft head over %lld tokens needs %.0f MiB of VRAM%s.\n",
                 (long long) n_tokens, mib(n_tokens), free_s.c_str());
    struct Subset { const char* name; int64_t tokens; const char* what; };
    static const Subset smaller[] = {{"cyrillic", 58963, "English, code and the Cyrillic script"},
                                     {"en", 40525, "English and code"}};
    std::string opts;
    for (const Subset& s : smaller)
        if (s.tokens < n_tokens) {
            char b[160];
            std::snprintf(b, sizeof(b), "%s--draft-vocab %s (%s, ~%.0f MiB)", opts.empty() ? "" : " or ", s.name,
                          s.what, mib(s.tokens));
            opts += b;
        }
    if (!opts.empty())
        std::fprintf(stderr, "strata mtp: hint: a smaller draft vocabulary needs less VRAM: %s. Start once with it - "
                             "START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                             "keeps it (\"draft_vocab\" in its strata-*.json config); or a smaller --context in "
                             "setup.\n",
                     opts.c_str());
    else
        std::fprintf(stderr, "strata mtp: hint: this is already the smallest shipped draft vocabulary: a smaller "
                             "--context (or closing what else uses the GPU) leaves it room.\n");
}
}  // namespace

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {
    const OnDevice on_device(device_);
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head (--native)"; return false; }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    // the draft head's token subset, when tools/draft_vocab.py wrote one
    if (dhead_ == nullptr) {
        std::vector<uint8_t> raw;
        if (read_file(rt_dir_ + "/draft_vocab.bin", raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            n_dvocab_ = (int64_t) (raw.size() / 4);
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            if (cudaMalloc((void**) &dvocab_, raw.size()) != cudaSuccess ||
                cudaMalloc((void**) &dhead_, (size_t) (n_dvocab_ * row_bytes)) != cudaSuccess) {
                err = "mtp: the draft head does not fit";
                draft_head_hint(n_dvocab_, row_bytes);
                return false;
            }
            cudaMemcpy(dvocab_, raw.data(), raw.size(), cudaMemcpyHostToDevice);
            strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_, nullptr);
            cudaDeviceSynchronize();
            vram_ += (uint64_t) (n_dvocab_ * row_bytes) + raw.size();
            std::fprintf(stderr, "strata mtp: draft head over %lld tokens (%.1f MiB)\n", (long long) n_dvocab_,
                         (double) (n_dvocab_ * row_bytes) / 1048576.0);
        }
    }
    if (coupled_draft_env() && cparams_ == nullptr && !setup_coupled(err)) return false;
    // ADAPTED (2026-10-09, THIS PORT): the staging buffer for the per-stage snapshots (STRATA_MTP_DUMP_STAGES).
    // Allocated HERE, on the host-side load path, because `record_forward` runs inside a stream capture and
    // `cudaMalloc` during capture is illegal - the first version did the lazy malloc there and the run died with
    // "iq_embed_rows: operation failed due to a previous error during capture".  Off unless the env var is set.
    if ((std::getenv("STRATA_MTP_DUMP_STAGES") != nullptr || std::getenv("STRATA_MTP_DUMP_TOKENS") != nullptr) &&
        g_stage_dev == nullptr && g_ != nullptr) {
        const size_t bytes = (size_t) 16 * 8192 * sizeof(float);   // 16 slots x 8192 floats (the wide attention slots)
        if (cudaMalloc((void**) &g_stage_dev, bytes) != cudaSuccess) {
            g_stage_dev = nullptr;
            std::fprintf(stderr, "strata mtp: the stage-snapshot buffer did not fit (%.1f KiB); stages stay off\n",
                         (double) bytes / 1024.0);
        }
    }
    // ADAPTED (2026-10-10, THIS PORT): one head-logit row (n_vocab floats), for the 0th hop's snapshot.  Allocated
    // here for the same reason as the stage buffer above: `record_forward` runs inside a stream capture.
    if (std::getenv("STRATA_MTP_DUMP_LOGITS") != nullptr && g_head_dev == nullptr && n_vocab_ > 0) {
        if (cudaMalloc((void**) &g_head_dev, (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
            g_head_dev = nullptr;
            std::fprintf(stderr, "strata mtp: the hop-logits buffer did not fit (%.1f KiB); the logits dump would "
                                 "read the LAST hop's row\n", (double) n_vocab_ * 4.0 / 1024.0);
        }
    }
    return true;
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    const int row0 = full ? step_row0 : -1 - step_row0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    // ADAPTED (2026-10-09, THIS PORT): STAGE SNAPSHOTS.  `record_forward` is RECORDED into a CUDA graph, so a host
    // copy here is illegal - these are plain D2D copies into a staging buffer, read out later by the debug block
    // (STRATA_MTP_DUMP_STAGES=<path>).  With them a reference implementation does not have to reproduce the whole
    // block before it can say anything: each stage is compared on its own, and the FIRST stage that differs is the
    // bug.  Slots: 0 = the seed (R_ right after the seed is built), 1 = post-attn-norm (mixed_), 2 = the
    // attention out-projection (bo_), 3 = the MLP out (y_), 4 = the head input (sample_), 5 = e2_ (the embedding
    // branch), 6 = h2_ (the hidden branch).  Slots 5/6 were added after the first comparison showed the seed is
    // close in DIRECTION to the reference (cos 0.74-0.83) but not equal: that isolates the disagreement to one
    // norm or one GEMM instead of leaving it "somewhere in the block".
    // NOT here: `cudaMalloc` is ILLEGAL inside a stream capture, and `record_forward` runs inside one.  The
    // staging buffer is allocated in `load` instead (see the note there), before any graph is captured.
    // Slots 10..13 are the ATTENTION'S OWN intermediates, in a wider slot (8192 floats = 4 x n_embd):
    //   10 = vcur_  (the raw V projection),  11 = qcur_ (q after per-head q_norm AND RoPE),
    //   12 = kcur_  (k after per-head k_norm AND RoPE),  13 = attn32_ (the attention output, pre o_proj).
    // With qcur_/kcur_ visible the RoPE question stops being indirect: the same vector can be produced here both
    // roped and unroped and compared, WITHOUT needing any cell's K/V (a 1-cell window makes q.k irrelevant to the
    // attention itself, which is why the first attempt could not see the rotation at all).
    // ADAPTED (2026-10-10, THIS PORT): ONE SLOT, ONE HOP.  `record_forward` runs the FULL layer once per hop -
    // the round's own row (`step_row0 == 2*max_t_-1`, the row the draft is judged on) and then one row per chain
    // step `j` (`step_row0 == max_t_+j-1`).  Every one of them used to write the SAME slot, so a slot read after
    // the round held the LAST hop's values, not the judged row's - which invalidated the "Rin_ is not the right
    // residual" reading of the previous round (see the ledger).  The guard below keeps only the 0th hop, which is
    // exactly the row every alignment check needs, and is a no-op for `--mtp-max-t 1` / window-1 runs where the
    // 0th hop is the only full pass there is (so the earlier single-hop results stand unchanged).
    const bool stage_hop = full && row0 == 2 * max_t_ - 1;
    auto stage_snap_n = [&](int slot, const float* src, int64_t n) {
        if (g_stage_dev == nullptr || !stage_hop) return;
        cudaMemcpyAsync(g_stage_dev + (size_t) slot * 8192, src, (size_t) n * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs);
    };
    auto stage_snap = [&](int slot, const float* src) {
        if (g_stage_dev == nullptr || !stage_hop) return;
        cudaMemcpyAsync(g_stage_dev + (size_t) slot * 8192, src,
                        (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, cs);
    };
    try {
        // ---- the two input branches
        const WeightRef* we = wt_->find("token_embd.weight");
        if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
        if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
            ne->gather_dev(tok_, T, emb_, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                 (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
        }
        // ADAPTED (2026-10-10, THIS PORT): WHICH IDS DOES THE GRAPH READ?  The post-hoc reads in the debug block
        // cannot answer it: `mtp_select` writes the selected token back INTO `tok_` in place (mtp.cpp:882 hands it
        // tok_ as both its input and its output), and 882 runs BEFORE the full pass at 888 - the pass whose stage
        // snapshots every alignment check so far has compared.  So the ids are snapshotted HERE, inside the
        // recorded graph, once per pass, into two halves of the otherwise unused slot 15:
        //   +0    the FULL pass (one row, the row the draft is judged on): int32 tok[0..7], then row_[0..3] at +8
        //   +4096 the K/V-only pass (T rows, the prompt's cells):            int32 tok[0..7], then row_[0..3] at +8
        // Read out under STRATA_MTP_DUMP_TOKENS beside the ids the HOST staged (h_tok_, mapped memory that the
        // device never writes), which is what separates "the host fed the wrong token" from "the graph read a
        // stale one".
        if (g_stage_dev != nullptr && (stage_hop || !full)) {
            const size_t half = (size_t) 15 * 8192 + (stage_hop ? 0 : 4096);
            const int nids = (int) (T < 8 ? T : 8);
            cudaMemcpyAsync(g_stage_dev + half, tok_, (size_t) nids * sizeof(int32_t), cudaMemcpyDeviceToDevice, cs);
            cudaMemcpyAsync(g_stage_dev + half + 8, row_, 4 * sizeof(int32_t), cudaMemcpyDeviceToDevice, cs);
        }
        // ADAPTED (2026-10-10, THIS PORT): THE PREFILL ROW'S K/V *BEFORE* THE CACHE.  `prefill`'s own e2_/h2_/
        // mixed_ were read back and match the formula at 0.99998 (_align29), yet the row that ends up in the draft
        // layer's K/V does not (K 0.91 / V 0.79, _align28).  So the divergence is between `mixed_` and the cache,
        // and the only two steps there are the k/v projections + k_norm + RoPE and `kv_append_*` itself.  The
        // K/V-only pass stages kcur_/vcur_ at slots 10/12 but the stage guard keeps them for the 0th hop, so this
        // keeps the KV-only pass's own copies in the free tail of slot 15 (after the ids and row records).
        if (g_stage_dev != nullptr && !full) {
            cudaMemcpyAsync(g_stage_dev + (size_t) 15 * 8192 + 4200, vcur_, (size_t) NKV * HD * sizeof(float),
                            cudaMemcpyDeviceToDevice, cs);
            cudaMemcpyAsync(g_stage_dev + (size_t) 15 * 8192 + 4720, kcur_, (size_t) NKV * HD * sizeof(float),
                            cudaMemcpyDeviceToDevice, cs);
        }
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(GGML_Q8_0, q8("fc_embedding.weight"), xq_, e2_, (int) N, (int) N, T, cs);
        stage_snap(5, e2_);   // the embedding branch on its own: one RMSNorm (enorm) + one GEMM
        if (mtp_plain()) {
            // the plain seed: ONE stream, produced the way the BODY produces its own LM-head input - the
            // same `output_hc_*` mixer, because in the donor's graph the MTP seed IS the post-norm hidden
            // state that feeds the LM head (llama.cpp qwen35moe: "post-norm hidden state feeds both the LM
            // head and the MTP seed below").  Summing the hc streams is NOT that vector.
            if (wt_ == nullptr) { err = "mtp: STRATA_MTP_PLAIN=1 needs the main model's weights bound first"; return false; }
            // Slots 7/8/9: the hidden branch's OWN input chain, snapshotted AT THE POINT OF CONSUMPTION.  The
            // post-hoc readings of Rin_ (the seed dump) are taken after the whole round has run, and `mtp_select`
            // writes into that same buffer; if the in-stream row 7 differs from the dumped one, the dump was the
            // stale side and the engine is innocent.  Slot 8 = xn_ (what lm_head_mix produced), slot 9 = hn_ (what
            // pre_fc_norm_hidden produced, i.e. the q8_1 input of fc_hidden).
            stage_snap(7, Rin_);
            // ADAPTED (2026-10-09, THIS PORT): `lm_head_mix`'s plain form normalises ONE row - the trunk calls it
            // once per row (verify.cpp walks the window row by row) - while THIS pass runs T rows: `record_forward`
            // is entered with T = the verify window for the K/V-only catch-up (step_row0 < 0), and that pass stages
            // the drafter's K/V for every cell of the window.  Called once, rows 1..T-1 of xn_ kept the previous
            // pass's contents and still went through `pre_fc_norm_hidden` and `fc_hidden` into the seed, so the
            // catch-up wrote K/V for T-1 of its T cells from a seed that was half stale.  Call it per row, exactly
            // the way the trunk does.
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss_->block;
                bb.R = Rin_ + (size_t) t * (size_t) (HC * N);
                bb.mixed = xn_ + (size_t) t * (size_t) N;
                if (!lm_head_mix(*wt_, g, bb, cs, err)) return false;
            }
            stage_snap(8, xn_);
            native_qsa_rms_norm_weighted(xn_, f32("pre_fc_norm_hidden.weight"), hn_, (int) N, T, EPS, cs);
            stage_snap(9, hn_);
            for (int c = 0; c < (int) HC; ++c) {   // the full layer runs one row at a time, so hn_ is T*N wide
                native_quantize_q8_1(hn_, xq_, (int) N, (int) T, cs);
                native_mmvq(GGML_Q8_0, q8("fc_hidden.weight"), xq_, h2_ + (size_t) c * N, (int) N, (int) N, (int) T, cs);
            }
            stage_snap(6, h2_);   // the hidden branch on its own: lm_head_mix (trunk output_norm) + hnorm + one GEMM
        } else {
            native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
            for (int c0 = 0; c0 < T * HC; c0 += 8) {
                const int nc = (int) std::min<int64_t>(8, T * HC - c0);
                native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
                native_mmvq(GGML_Q8_0, q8("fc_hidden.weight"), xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
            }
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        stage_snap(0, R_);   // the seed, before anything reads it
        // ---- the attention hyper-connection
        if (mtp_plain()) {
            // the plain donor's pre-attention norm (attn_norm) is an ordinary RMSNorm over the one stream
            native_qsa_rms_norm_weighted(R_, f32("attn_hyper_connection.hc_norm.weight"), mixed_, (int) N, T, EPS, cs);
        } else {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        stage_snap(1, mixed_);   // post-attn-norm, both paths converge here
        // ---- attention: K/V into the layer's own cache, then (full) dense attention over every cell
        auto norm_rope = [&](float* data, const float* gamma, int rows, int cols, const int32_t* p) {
            native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
            if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
            else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st_.cos_tab, st_.sin_tab, p, cs);
        };
        native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.k_proj.weight"), xq_, kcur_, (int) N, (int) (NKV * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.v_proj.weight"), xq_, vcur_, (int) N, (int) (NKV * HD), T, cs);
        stage_snap_n(10, vcur_, T * NKV * HD);   // the raw V projection, before it goes into the cache
        for (int t = 0; t < T; ++t) {
            norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH);
            if (st_.kv_rot) {   // rotated K and V (kv_q4.hpp): Q4_0, and INT8 with STRATA_KV_ROT=1
                fwht256_inplace_cuda(kcur_ + t * NKV * HD, NKV, cs);
                fwht256_inplace_cuda(vcur_ + t * NKV * HD, NKV, cs);
            }
            // stored in the state's own format (#293 appended rotated INT8 K/V as Q4_0, into pools INT8 never has)
            if (st_.kv_q4)
                kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                  vcur_ + t * NKV * HD, s, cs, &st_.host);
            else if (st_.kv_int8)
                kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                  kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
            else
                kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                               vcur_ + t * NKV * HD, s, cs, &st_.host);
        }
        stage_snap_n(12, kcur_, T * NKV * HD);   // k after per-head k_norm AND RoPE
        if (!full) return true;
        native_mmvq(GGML_Q8_0, q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        stage_snap_n(14, qfull_, T * NH * 2 * HD);   // [q(256)|gate(256)] per head, before the split
        for (int t = 0; t < T; ++t) {
            float* qc = qcur_ + t * NH * HD;
            if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qc, f32("self_attn.q_norm.weight"), (int) NH, (int) HD, pos + t * NH);
            if (st_.kv_rot) fwht256_inplace_cuda(qc, NH, cs);
        }
        stage_snap_n(11, qcur_, T * NH * HD);   // q after per-head q_norm AND RoPE: the direct RoPE witness
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
        if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        for (int t = 0; t < T; ++t)
            native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD, (int) NH, (int) HD, cs);
        stage_snap_n(13, attn32_, T * NH * HD);   // the attention output, before o_proj
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.o_proj.weight"), xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        stage_snap(2, bo_);   // the attention's write, before it is folded into the residual
        // ---- the MLP hyper-connection (the attention write folded in)
        if (mtp_plain()) {
            add_inplace(R_, bo_, (int64_t) T * N, cs);   // the attention write, into the one stream
            native_qsa_rms_norm_weighted(R_, f32("mlp_hyper_connection.hc_norm.weight"), mixed_, (int) N, T, EPS, cs);
        } else {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- MoE: router, the 512 resident experts, the shared expert, the combine, the write
        for (int t = 0; t < T; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * ne_, (int) N, (int) ne_, cs);
            // ADAPTED: the native fused router is canonical-512x10 ONLY - the same guard layer.cpp carries.
        // The draft layer takes the generic kernel unless it is exactly that shape; without this guard
        // native_router_top10 refuses the buffers ("requires a stream, aligned spans, and disjoint
        // outputs") and the whole run exits 1.
        if (native_router_enabled() && ne_ == 512 && K == 10) native_router_top10(logits_ + t * ne_, ids_ + t * K, w_ + t * K, cs);
            else router_top10(logits_ + t * ne_, 1, (int) ne_, (int) K, ids_ + t * K, w_ + t * K, cs);
        }
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) strata::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = GGML_Q8_0; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
        nsw.up_type = GGML_Q8_0; nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
        nsw.down_type = GGML_Q8_0; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
        nsw.q8_1 = xq_;
        const SForm none{};
        const bool need_bf16_x = !shared_expert_native_bf16_enabled();
        static const bool fuse_head_gr = [] {
            const char* v = std::getenv("STRATA_FUSE_HEAD_GR");
            return v != nullptr && std::atoi(v) != 0;
        }();
        for (int t = 0; t < T; ++t) {
            if (need_bf16_x) f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, cs);
            shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                          nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_, shared_ + t * N,
                          N, g.n_ff, 32, cs, mixed_ + t * N, &nsw);
            if (native_moe_combine_enabled())
                native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            else
                moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            if (!fuse_head_gr && !mtp_plain())
                gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        if (mtp_plain()) add_inplace(R_, y_, (int64_t) T * N, cs);   // the MLP write, into the one stream
        stage_snap(3, y_);   // the MoE's combined write (routed + shared), before it is folded in
        // ---- the final mixer and the main model's head
        if (mtp_plain()) {
            // the plain donor's head norm (nextn.shared_head_norm) over the one stream, then the head
            native_qsa_rms_norm_weighted(R_, f32("hyper_connection_mixer.hc_norm.weight"), sample_, (int) N, T, EPS, cs);
        } else if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = y_ + t * N; fa[t].inj_prev = inj2_ + t * HC;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].w_inject = nullptr;
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = dummy_inj_; fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else {
            for (int t = 0; t < T; ++t)
                gr_read(R_ + (size_t) t * HC * N, f32("hyper_connection_mixer.hc_norm.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, EPS, gs, ss.block.gr,
                        sample_ + t * N, dummy_inj_, cs);
        }
        stage_snap(4, sample_);   // the head's input, after the final norm
        // ADAPTED (2026-10-10, THIS PORT): STRATA_MTP_SKIP_BLOCK=1 - a DIAGNOSTIC, not a feature.  The block is
        // now verified end to end: an independent full-chain run from (h, token) reproduces the engine's own head
        // input at cos 0.999805 (_align32), and the reference's graph body matches ours line by line
        // (qwen35moe.cpp:604-693).  Yet the draft distribution it produces shares NOTHING with the target's at the
        // same cell (top-10 overlap 0/10, the answer ranked 45k-94k), while llama.cpp accepts 8 of 10 drafts on
        // this very prompt.  So the one thing left to separate is the block's TRANSFORM from its INPUTS: with this
        // env set the head input is replaced by `xn_` (the h the pass was handed, already output_norm'd - i.e.
        // exactly what the target's own head consumes).  If acceptance then jumps, the plumbing, the h and the
        // head are all right and the block's transform is what destroys the draft; if it stays at zero, the h is
        // not the quantity the reference feeds.
        if (std::getenv("STRATA_MTP_SKIP_BLOCK") != nullptr) {
            cudaMemcpyAsync(sample_, xn_, (size_t) T * (size_t) N * sizeof(float), cudaMemcpyDeviceToDevice, cs);
        }
        native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const int64_t nv = sub ? n_dvocab_ : n_vocab_;
        native_mmvq(head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        // ADAPTED (2026-10-10, THIS PORT): keep the 0th hop's row (see g_head_dev).  A device->device copy inside
        // the recorded graph is legal (only cudaMalloc is not) and costs one row per round.
        if (g_head_dev != nullptr && stage_hop) {
            cudaMemcpyAsync(g_head_dev, head_logits_, (size_t) nv * sizeof(float), cudaMemcpyDeviceToDevice, cs);
        }
        if (coupled_rec_) {
            // coupled draft sampling: the target's chain and Philox draw for the row that will verify this draft
            // (counter = this cell + 1, from its step record), penalties over the ring; T == 1 (full layer)
            coupled_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                 cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, cs);
            return true;
        }
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
        if (sub) map_ids(out_ids_, dvocab_, T, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}  // namespace

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    if (prefill_exec_[T]) return true;
    using namespace strata::kernels;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}

bool MtpDrafter::capture_prefill_dev(int T, std::string& err) {
    if (prefill_dev_exec_[T]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T); tok_/step_/pos_ filled before launch
    return finish_capture(cs_, ok, prefill_dev_exec_[T], "prefill (device inputs)", err);
}

bool MtpDrafter::capture_round(int T, bool coupled, std::string& err) {
    cudaGraphExec_t& exec = coupled ? round_exec_c_[T] : round_exec_[T];
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    bool ok = true;
    // coupled: the request's chain and the penalty history's base, for this round's drafts
    if (coupled) coupled_draft_stage(m_cparams_, m_chist_, cparams_, cring_, kCoupledHistCap, cs_);
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) 2 * T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) 2 * T * g_->n_head, cs_);
    copy_i32_from_mapped(row_, m_row_, 2, cs_);
    copy_from_mapped(Rin_, window_R_, (int64_t) T * HCN, cs_);
    // the catch-up: K/V for the window's T cells, then the full layer for row a only (its cell's K/V is written
    // again, identically), staged by the host in step row 2*max_t - 1; the draft chain is one graph per step
    // (`capture_step`) so the host can stop it when a draft is unlikely
    const int ra = 2 * max_t_ - 1;
    ok = record_forward(T, -1, cs_, err);
    if (ok) mtp_select(Rin_, HCN, tok_, row_, Rin_, tok_, nullptr, 0, cs_);
    if (ok) {
        copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
        copy_i32_from_mapped(pos_ + ra * g_->n_head, m_pos_ + ra * g_->n_head, g_->n_head, cs_);
        coupled_rec_ = coupled;
        coupled_j_ = 0;
        ok = record_forward(1, ra, cs_, err);
        coupled_rec_ = false;
    }
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
    return finish_capture(cs_, ok, exec, coupled ? "round (coupled)" : "round", err);
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, bool coupled, std::string& err) {
    cudaGraphExec_t& exec = coupled ? step_exec_c_[j] : step_exec_[j];
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
    coupled_rec_ = coupled;
    coupled_j_ = j;
    bool ok = record_forward(1, row, cs_, err);
    coupled_rec_ = false;
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
    return finish_capture(cs_, ok, exec, coupled ? "step (coupled)" : "step", err);
}

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    strata::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cudaStreamSynchronize(cs_);
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // ADAPTED (2026-10-10, THIS PORT): WHAT THIS PASS ACTUALLY COMPUTED, read back OUTSIDE the graph.  The cache
    // instrument (STRATA_MTP_DUMP_KV) showed that the rows this pass writes into the draft layer's K/V disagree
    // with the composition of the very inputs it was handed - K 0.91 / V 0.79 on a single-row run (2-id prompt),
    // while the draft round's own row, same code and same weights, matches at 0.99998.  `record_forward` runs the
    // same branch for both passes (`if (!full) return true;` sits AFTER the K/V append), so the divergence has to
    // live in one of the staged buffers.  `prefill` is a plain host function, so a device->host copy here is legal,
    // and it names the branch: the embedding side e2_, the hidden side h2_, their attn_norm output mixed_, or their
    // sum R_ (the seed).
    // STRATA_MTP_DUMP_PREFILL_ROW=<path>: int32 {chunk_c, T, cell0, n}, then T*N floats each of e2_, h2_, mixed_, R_.
    int64_t pf_c = -1;
    int pf_T = 0;
    auto dump_prefill_rows = [&]() {
        const char* path = std::getenv("STRATA_MTP_DUMP_PREFILL_ROW");
        if (path == nullptr || pf_T <= 0 || g_ == nullptr) return;
        const size_t nf = (size_t) pf_T * (size_t) g_->n_embd;
        std::vector<float> e2(nf), h2(nf), mx(nf), sd(nf);
        const bool ok = cudaMemcpy(e2.data(), e2_, nf * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(h2.data(), h2_, nf * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(mx.data(), mixed_, nf * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(sd.data(), R_, nf * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        std::FILE* f = ok ? std::fopen(path, "ab") : nullptr;
        if (f != nullptr) {
            const int32_t hdr[4] = {(int32_t) pf_c, pf_T, (int32_t) cell0, (int32_t) n};
            std::fwrite(hdr, sizeof(hdr), 1, f);
            std::fwrite(e2.data(), 4, nf, f);
            std::fwrite(h2.data(), 4, nf, f);
            std::fwrite(mx.data(), 4, nf, f);
            std::fwrite(sd.data(), 4, nf, f);
            std::fclose(f);
        }
        std::fprintf(stderr, "mtp prefillrows: chunk_c=%lld T=%d e2[0]=%.6g h2[0]=%.6g mixed[0]=%.6g seed[0]=%.6g %s\n",
                     (long long) pf_c, pf_T, nf > 0 ? e2[0] : 0.0f, nf > 0 ? h2[0] : 0.0f,
                     nf > 0 ? mx[0] : 0.0f, nf > 0 ? sd[0] : 0.0f, f != nullptr ? "written" : "FAILED");
        std::fflush(stderr);
    };
    // ADAPTED (2026-10-09, THIS PORT): THE PROMPT-CELL SEEDS, at the entry of the pass that consumes them.  This
    // function is what builds the drafter's OWN K/V for the prompt's cells - the K/V the draft's attention reads
    // when its window is the real one (mtp_window defaults to 32768 = every cell).  Everything verified so far was
    // verified with --mtp-window 1, where the draft attends to its own cell alone, so THIS path has never been
    // checked.  Dumped at the entry rather than inside the group loop because `prefill` has two paths (E-4 by
    // default, the old per-group loop under STRATA_MTP_PREFILL_SYNC=1) and the arguments hold every cell either way.
    // STRATA_MTP_DUMP_PREFILL=<path>: int32 {n, cell0, HCN}, then n int32 tokens, then n*HCN floats.
    if (const char* pref_dump = std::getenv("STRATA_MTP_DUMP_PREFILL"); pref_dump != nullptr && n > 0) {
        std::vector<float> rr((size_t) n * (size_t) HCN);
        if (cudaMemcpy(rr.data(), R_rows, rr.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::FILE* pf = std::fopen(pref_dump, "ab");
            if (pf != nullptr) {
                const int32_t hdr[3] = {(int32_t) n, (int32_t) cell0, (int32_t) HCN};
                std::fwrite(hdr, sizeof(hdr), 1, pf);
                std::fwrite(next_tokens, sizeof(int32_t), (size_t) n, pf);
                std::fwrite(rr.data(), sizeof(float), rr.size(), pf);
                std::fclose(pf);
                std::fprintf(stderr, "mtp prefilldump: n=%lld cell0=%lld HCN=%lld\n",
                             (long long) n, (long long) cell0, (long long) HCN);
            }
        }
        std::fflush(stderr);
    }
    // cells the window can never reach again need no K/V
    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    // E-4: every group's token / step / position records uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end (a group of <= max_t rows used to be staged in mapped
    // memory and synced before the next: ~5,500 host round trips on a 32K prompt).  The same work in the same
    // order.  STRATA_MTP_PREFILL_SYNC=1 keeps the old loop, =0 forces E-4.
    // HIP defaults to the old loop.  This pass runs whenever E-9 (Prefill::draft_kv) declines, which it does for the
    // drafter's ring (KV streaming, --kv-resident), and on gfx1201 / ROCm 7.2.4 the E-4 queue (a graph launch and four
    // device copies per group, ~2,000 groups per 8192-token chunk, no sync) sometimes never completes: the prompt hangs
    // in hipGraphLaunch or the final sync until the watchdog ends the engine.  R9700, full IQ3_XXS, --kv-resident
    // 32768, mixed load (chats, 32K and 7K prompts, deep follow-ups): E-4 hung in the first round on 2 of 2 tries,
    // the old loop ran 30 of 30 rounds clean, prompt speed unchanged.
    static const bool per_group_sync = [] {
        if (const char* v = std::getenv("STRATA_MTP_PREFILL_SYNC")) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
        return true;
#else
        return false;
#endif
    }();
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (!per_group_sync && n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) cudaFree(pf_dev_);
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(pf_dev_, rec.data(), rec.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        for (int64_t c = 0; c < n; c += max_t_) {
            const int T = (int) std::min<int64_t>(max_t_, n - c);
            if (cell0 + c + T <= first_needed) continue;
            if (!capture_prefill_dev(T, err)) return false;
            pf_c = c; pf_T = T;   // the rows the dump below reads back (this chunk's)
            if (cudaMemcpyAsync(tok_, d_tk + c, (size_t) T * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(step_, d_stp + c * 4, (size_t) T * 16, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(pos_, d_ps + c * NHp, (size_t) (T * NHp) * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                                cs_) != cudaSuccess ||
                cudaGraphLaunch(prefill_dev_exec_[T], cs_) != cudaSuccess) {
                err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        dump_prefill_rows();
        ms_prefill += ms_since(t0);
        return true;
    }
    for (int64_t c = 0; c < n; c += max_t_) {
        const int T = (int) std::min<int64_t>(max_t_, n - c);
        if (cell0 + c + T <= first_needed) continue;
        if (!capture_prefill(T, err)) return false;
        pf_c = c; pf_T = T;   // the rows the dump below reads back (this chunk's)
        for (int t = 0; t < T; ++t) {
            const int64_t cell = cell0 + c + t;
            h_tok_[t] = next_tokens[c + t];
            h_step_[t * 4 + 0] = (int32_t) cell;
            h_step_[t * 4 + 1] = (int32_t) (cell + 1);
            h_step_[t * 4 + 2] = (int32_t) ((cell + 1) / 4);
            h_step_[t * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < g_->n_head; ++h) h_pos_[t * g_->n_head + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                            cs_) != cudaSuccess ||
            cudaGraphLaunch(prefill_exec_[T], cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    dump_prefill_rows();
    ms_prefill += ms_since(t0);
    return true;
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
    const bool cp = coupled_active_;   // coupled draft sampling for this request: its own graphs
    if (!capture_round(T, cp, err)) return false;
    // ADAPTED 2026-10-09: the window consumes drafts[1..T-1] (drafts[0] predicts the token we already sampled
    // and is dropped), so the drafter must produce max_t_ hops, not max_t_-1.  See generate.cpp:8231.
    const int max_steps = std::min(max_t_, max_drafts_);
    for (int j = 1; j < max_steps; ++j)
        if (!capture_step(j, cp, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, coupled_draft_cell(p, a, 0));   // p + a: draft 0's cell
    for (int j = 1; j < max_steps; ++j)
        put(max_t_ + j - 1, coupled_draft_cell(p, a, j));
    for (int j = 0; j < max_steps; ++j) ((volatile int32_t*) h_out_)[j] = -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // ADAPTED diagnostic: stamp parts_[0] with a sentinel BEFORE the round runs.  If the stamp survives the
    // round, the grouped MoE never wrote the buffer (so a huge value read there is stale memory, not math).
    static const bool dbg_stamp = [] { const char* e = std::getenv("STRATA_MTP_DEBUG_DRAFTS"); return e != nullptr && e[0] == '1'; }();
    if (dbg_stamp && parts_ != nullptr) {
        const float sv = -12345.0f;
        cudaMemcpy(parts_, &sv, sizeof(float), cudaMemcpyHostToDevice);
    }

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;
    const bool do_ple = pss != nullptr && pss->ple.ready() && pss->ple.table != nullptr;
    int32_t ple_prev[2] = {do_ple ? pss->ple_prev[0] : 0, do_ple ? pss->ple_prev[1] : 0};
    auto prefetch_ple = [&](int32_t tok) {
        if (!do_ple || tok < 0) return;
        uint32_t rows16[strata::kernels::PLE_N_HEADS];
        strata::kernels::ngram_rows(&tok, ple_prev, 1, pss->ple.consts, rows16);
        ple_prev[0] = ple_prev[1];
        ple_prev[1] = tok;
        pss->ple.table->prefetch_rows(rows16);
    };

    int n = 0;
    if (min_p <= 0.0f && max_steps > 0) {
        if (cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        for (int j = 1; j < max_steps; ++j) {
            if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        (void) cudaStreamQuery(cs_);
        prefetch_ple(tokens[a]);
        for (int j = 0; j + 1 < max_steps; ++j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                _mm_pause();
#endif
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            if (probs) probs[j] = ((volatile float*) h_prob_)[j];
            prefetch_ple(drafts[j]);
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        const int last = max_steps - 1;
        drafts[last] = ((volatile int32_t*) h_out_)[last];
        if (probs) probs[last] = ((volatile float*) h_prob_)[last];
        prefetch_ple(drafts[last]);
        n = max_steps;
    } else if (max_steps > 0) {
        if (cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        (void) cudaStreamQuery(cs_);
        prefetch_ple(tokens[a]);
        auto wait_step = [&](int j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                _mm_pause();
#endif
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
        };
        wait_step(0);
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pj = ((volatile float*) h_prob_)[0];
        if (probs) probs[0] = pj;
        n = 1;
        for (int j = 1; j < max_steps && pj >= min_p; ++j) {
            if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            (void) cudaStreamQuery(cs_);
            prefetch_ple(drafts[j - 1]);
            wait_step(j);
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            if (probs) probs[j] = pj;
            ++n;
        }
        prefetch_ple(drafts[n - 1]);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    // ADAPTED: STRATA_MTP_DEBUG_DRAFTS=1 prints the first rounds' answer (the body's own token at the draft's
    // cell) against the draft and the draft's own probability - the readout that separates "the block computes
    // garbage" from "the block is fine but trained on another body".  Off by default; never in a captured graph.
    static const bool dbg_drafts = [] { const char* e = std::getenv("STRATA_MTP_DEBUG_DRAFTS"); return e != nullptr && e[0] == '1'; }();
    if (dbg_drafts) {
        static int shown = 0;
        if (shown < 3) {
            ++shown;
            // ADAPTED (2026-10-10, THIS PORT): the label is `seed_tok`, not `answer`.  `tokens` now carries the
            // WINDOW'S INPUT ids (generate.cpp passes `window`, not `outv`): the engine's seed pairs the residual
            // at cell p+a with the token AT that cell (window[a] = x_{p+a}), which is what the reference does
            // (speculative.cpp:1624-1625 pairs `pending_h` with `id_last` at `pos0`).  What the draft must predict
            // is the caller's `outv[a]` = x_{p+a+1} - one step ahead of this token - and that is compared where the
            // window is verified (generate.cpp: `window[a+1] == outv[a]`).
            std::fprintf(stderr, "mtp debug: cell=%lld row=%d seed_tok=%d n=%d", (long long) p, a, tokens[a], n);
            for (int j = 0; j < n && j < 4; ++j) std::fprintf(stderr, "  draft[%d]=%d p=%.4f", j, drafts[j], probs ? probs[j] : -1.0f);
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
            // which stage first goes non-finite: the seed, each block stage, the head input, the logits.
            // These copies are legal here: the graphs have been launched and the stream is idle (outside capture).
            {
                const float* bufs[] = {Rin_, xn_, hn_, mixed_, bo_, y_, sample_, head_logits_};
                const char* nm[] = {"Rin", "seed", "hnorm", "mixed", "attn", "moe", "headin", "logits"};
                std::fprintf(stderr, "mtp debug nan:");
                for (int i = 0; i < 8; ++i) {
                    float v = 0.0f;
                    cudaMemcpy(&v, bufs[i], sizeof(float), cudaMemcpyDeviceToHost);
                    const bool bad = !(v == v) || v > 3.0e38f || v < -3.0e38f;
                    std::fprintf(stderr, "  %s=%.4g%s", nm[i], v, bad ? "!NONFINITE" : "");
                }
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
                // the seed's SOURCE (the verify window's rows) against the device copy and the session's live
                // residual: this separates "those rows are already garbage" from "the copy is wrong".
                // window_R_ is DEVICE memory - a host dereference of it faults, so every read goes through CUDA.
                {
                    float w0 = 0.0f, r0 = 0.0f, s0 = 0.0f;
                    if (window_R_ != nullptr) cudaMemcpy(&w0, window_R_, sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&r0, Rin_, sizeof(float), cudaMemcpyDeviceToHost);
                    if (ss_->block.R != nullptr) cudaMemcpy(&s0, ss_->block.R, sizeof(float), cudaMemcpyDeviceToHost);
                    std::fprintf(stderr, "mtp debug seed-src: windowR[0]=%.6g  devRin[0]=%.6g  sessionR[0]=%.6g\n", w0, r0, s0);
                    std::fflush(stderr);
                }
                // ADAPTED (2026-10-09, THIS PORT): the draft's OWN distribution for this row, dumped raw.
                // `p=%.4f` above is the draft's probability of ITS OWN pick; it cannot answer the question that
                // decides this whole line - is the draft standing on the token the body actually produces for this
                // cell (`the answer`, which is only known after the fact and appears on the same log's
                // `output :` line), ranked second or third, or is it on an unrelated token altogether?  Raw
                // float32 rows, appended, one header line per row on stderr; the lookup is offline.
                // STRATA_MTP_DUMP_LOGITS=<path>.  Off by default; never inside a captured graph.
                static const char* dump_logits = std::getenv("STRATA_MTP_DUMP_LOGITS");
                if (dump_logits != nullptr) {
                    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
                    std::vector<float> row((size_t) nv);
                    // the 0th hop's row when the snapshot buffer exists (same hop as every other staged slot),
                    // otherwise the last hop's row - shouted about below so a reading can never be mis-attributed
                    const float* lsrc = g_head_dev != nullptr ? g_head_dev : head_logits_;
                    if (cudaMemcpy(row.data(), lsrc, (size_t) nv * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                        std::fprintf(stderr, "mtp dump: the logits row copy failed\n");
                    } else {
                        std::FILE* f = std::fopen(dump_logits, "ab");
                        if (f == nullptr) {
                            std::fprintf(stderr, "mtp dump: cannot open %s\n", dump_logits);
                        } else {
                            const long long off = (long long) std::ftell(f);
                            const size_t wrote = std::fwrite(row.data(), sizeof(float), row.size(), f);
                            std::fclose(f);
                            std::fprintf(stderr, "mtp dump: cell=%lld row=%d nv=%lld sub=%d off=%lld wrote=%zu\n",
                                         (long long) p, a, (long long) nv, dhead_ != nullptr ? 1 : 0, off, wrote);
                        }
                    }
                    std::fflush(stderr);
                }
                // ADAPTED (2026-10-09, THIS PORT): the SEED buffer and the block's own output for this draft, next
                // to the logits above.  `Rin` in the nan print below is element 0 of a buffer that is REUSED between
                // rows, so it cannot say WHICH row the full pass actually consumed - and that is the one thing a
                // row-by-row reference implementation of this block needs.  Per drafted round it writes:
                //   int32 {cell, row, sel0, sel1}, int32 {tok0, tok1},
                //   2*HC*N floats of Rin_ (its first two rows), N floats of R_ (the block's residual out),
                //   N floats of sample_ (the head's input).
                // STRATA_MTP_DUMP_SEED=<path>.  Off by default; never inside a captured graph.
                static const char* dump_seed = std::getenv("STRATA_MTP_DUMP_SEED");
                if (dump_seed != nullptr) {
                    const int64_t Ns = g_->n_embd, HCs = g_->hc;
                    std::vector<float> s_in((size_t) (2 * HCs * Ns)), s_out((size_t) Ns), s_mix((size_t) Ns);
                    int32_t sel[2] = {0}, tks[2] = {0};
                    cudaMemcpy(s_in.data(), Rin_, s_in.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(s_out.data(), R_, s_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(s_mix.data(), sample_, s_mix.size() * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(sel, row_, sizeof(sel), cudaMemcpyDeviceToHost);
                    cudaMemcpy(tks, tok_, sizeof(tks), cudaMemcpyDeviceToHost);
                    std::FILE* gs = std::fopen(dump_seed, "ab");
                    if (gs == nullptr) {
                        std::fprintf(stderr, "mtp seeddump: cannot open %s\n", dump_seed);
                    } else {
                        const int32_t hdr[4] = {(int32_t) p, (int32_t) a, sel[0], sel[1]};
                        std::fwrite(hdr, sizeof(hdr), 1, gs);
                        std::fwrite(tks, sizeof(tks), 1, gs);
                        std::fwrite(s_in.data(), sizeof(float), s_in.size(), gs);
                        std::fwrite(s_out.data(), sizeof(float), s_out.size(), gs);
                        std::fwrite(s_mix.data(), sizeof(float), s_mix.size(), gs);
                        std::fclose(gs);
                        std::fprintf(stderr, "mtp seeddump: cell=%lld row=%d sel=%d,%d tok=%d,%d\n",
                                     (long long) p, a, sel[0], sel[1], tks[0], tks[1]);
                    }
                    std::fflush(stderr);
                }
                // ADAPTED (2026-10-09, THIS PORT): read the per-stage snapshots out (taken inside the recorded
                // graph by `stage_snap`): 5 x n_embd floats = {seed, post-attn-norm, attn out-proj, MoE out,
                // head input}, in that order, appended once per drafted round.  Same env var as the producer:
                // STRATA_MTP_DUMP_STAGES=<path>.
                if (const char* dump_stages = std::getenv("STRATA_MTP_DUMP_STAGES"); dump_stages != nullptr && g_stage_dev != nullptr) {
                    const size_t one = (size_t) g_->n_embd;
                    std::vector<float> st((size_t) 16 * 8192);
                    if (cudaMemcpy(st.data(), g_stage_dev, st.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess) {
                        std::FILE* ht = std::fopen(dump_stages, "ab");
                        if (ht != nullptr) {
                            const int32_t hdr2[2] = {(int32_t) p, (int32_t) a};
                            std::fwrite(hdr2, sizeof(hdr2), 1, ht);
                            std::fwrite(st.data(), sizeof(float), st.size(), ht);
                            std::fclose(ht);
                            std::fprintf(stderr, "mtp stageout: cell=%lld n=%zu\n", (long long) p, st.size());
                        }
                        std::fflush(stderr);
                    }
                }
                // ADAPTED (2026-10-10, THIS PORT): THE h THE SEED ACTUALLY GETS.  `Rin_` is filled by
                // `copy_from_mapped(Rin_, window_R_, T*HCN)` (mtp.cpp:876) - a kernel that reads `src` as
                // `const volatile float4*`, i.e. it was written for HOST-mapped memory, while `window_R_` is a
                // DEVICE pointer (`ver.final_R_all()` snapshotted at bind()).  With UVA the kernel still works, so
                // a stale or mis-resolved pointer is silent - and a stale h would leave the draft head with no
                // context, which is exactly what the last measurement showed (`output.weight @ xn_` ranks the
                // target's own answer at 80,954 of 248,320, while by construction the verifier's argmax over the
                // same vector IS that answer).  So print both: the h the seed received, and the h's source.
                if (std::getenv("STRATA_MTP_DUMP_TOKENS") != nullptr) {
                    float wr[4] = {0.0f}, rr[4] = {0.0f};
                    if (window_R_ != nullptr) cudaMemcpy(wr, window_R_, sizeof(wr), cudaMemcpyDeviceToHost);
                    cudaMemcpy(rr, Rin_, sizeof(rr), cudaMemcpyDeviceToHost);
                    std::fprintf(stderr, "mtp hsrc: windowR=%.6g %.6g %.6g %.6g | Rin=%.6g %.6g %.6g %.6g | %s\n",
                                 wr[0], wr[1], wr[2], wr[3], rr[0], rr[1], rr[2], rr[3],
                                 (window_R_ != nullptr && wr[0] == rr[0]) ? "equal" : "DIFFERENT");
                    std::fflush(stderr);
                }
                // ADAPTED (2026-10-10, THIS PORT): THE K/V THE ATTENTION ACTUALLY READS.  Rebuilding the context
                // rows from the composition reproduces the engine's attention at cos 0.899 (ledger #225-#228) and
                // everything else about them is verified - composition branch by branch (0.9998/0.99998), the gate
                // convention, the head mapping, the own row, and quantisation (an fp16-KV arm gave BIT-IDENTICAL
                // drafts).  So the residue has to be in what the cache holds, and the only way to see it is to ask
                // the engine's own gather for the rows it would read: `kv_gather_step` (fp16 pools) or
                // `kv_gather_q8_step` (int8 pools), ids = 0..cell, count from the query row's own step record.
                // STRATA_MTP_DUMP_KV=<path>: per drafted round, int32 {cell, a, cells, per_row}, then `cells *
                // per_row` fp16 K values, then the same for V (row-major [cell][kv_head][head_dim]).
                if (const char* kvdump = std::getenv("STRATA_MTP_DUMP_KV"); kvdump != nullptr) {
                    const strata::kernels::QsaAttnPools px = qsa_attn_pools(st_);
                    const int64_t cells = (int64_t) p + 1;
                    const int64_t row = (int64_t) g_->n_head_kv * (int64_t) g_->head_dim;
                    const size_t bytes = (size_t) cells * (size_t) row * sizeof(uint16_t);
                    std::vector<uint16_t> hk(bytes / 2), hv(bytes / 2);
                    uint16_t* dk = nullptr; uint16_t* dv = nullptr;
                    int32_t* dids = nullptr; int32_t* dst = nullptr;
                    bool okv = cudaMalloc((void**) &dk, bytes) == cudaSuccess &&
                               cudaMalloc((void**) &dv, bytes) == cudaSuccess &&
                               cudaMalloc((void**) &dids, (size_t) cells * 4) == cudaSuccess &&
                               cudaMalloc((void**) &dst, 16) == cudaSuccess;
                    if (okv && cells > 0) {
                        std::vector<int32_t> ids((size_t) cells);
                        for (int64_t i = 0; i < cells; ++i) ids[(size_t) i] = (int32_t) i;
                        int32_t st4[4] = {0};
                        cudaMemcpy(st4, step_ + (2 * max_t_ - 1) * 4, sizeof(st4), cudaMemcpyDeviceToHost);
                        cudaMemcpy(dids, ids.data(), (size_t) cells * 4, cudaMemcpyHostToDevice);
                        cudaMemcpy(dst, st4, sizeof(st4), cudaMemcpyHostToDevice);
                        const strata::kernels::QsaShapes sh = shapes_of(*g_);
                        const char* how = "none";
                        if (px.k_q != nullptr && px.v_q != nullptr) {
                            strata::kernels::kv_gather_q8_step(px.k_q, px.v_q, px.k_scale, px.v_scale,
                                                               px.page_table, dids, dst, cells, sh, dk, dv, cs_);
                            how = "int8";
                        } else if (px.k_pool != nullptr && px.v_pool != nullptr) {
                            strata::kernels::kv_gather_step(px.k_pool, px.v_pool, px.page_table, dids, dst, cells,
                                                            sh, dk, dv, cs_);
                            how = "fp16";
                        }
                        okv = cudaStreamSynchronize(cs_) == cudaSuccess &&
                              cudaMemcpy(hk.data(), dk, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                              cudaMemcpy(hv.data(), dv, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
                        std::FILE* kf = okv ? std::fopen(kvdump, "ab") : nullptr;
                        if (kf != nullptr) {
                            const int32_t hdr[4] = {(int32_t) p, a, (int32_t) cells, (int32_t) row};
                            std::fwrite(hdr, sizeof(hdr), 1, kf);
                            std::fwrite(hk.data(), 1, bytes, kf);
                            std::fwrite(hv.data(), 1, bytes, kf);
                            std::fclose(kf);
                        }
                        std::fprintf(stderr, "mtp kvdump: cell=%lld rows=%lld per_row=%lld src=%s step=[%d,%d,%d,%d] %s\n",
                                     (long long) p, (long long) cells, (long long) row, how,
                                     st4[0], st4[1], st4[2], st4[3], kf != nullptr ? "written" : "FAILED");
                    } else {
                        std::fprintf(stderr, "mtp kvdump: allocation failed\n");
                    }
                    if (dk) cudaFree(dk);
                    if (dv) cudaFree(dv);
                    if (dids) cudaFree(dids);
                    if (dst) cudaFree(dst);
                    std::fflush(stderr);
                }
                // ADAPTED (2026-10-10, THIS PORT): the ids the HOST staged against the ids the GRAPH read.  Slot 15
                // is written inside BOTH recorded passes (see the note in record_forward); h_tok_ is mapped host
                // memory that the device never writes back, so it is the un-polluted record of the host's intent.
                // Per drafted round: one stderr line and one appended record
                //   int32 {cell, a, T, nd} then int32 host[8], int32 full[12], int32 kv[12].
                // Off by default; never inside a captured graph.
                if (const char* tokdump = std::getenv("STRATA_MTP_DUMP_TOKENS"); tokdump != nullptr && g_stage_dev != nullptr) {
                    const int nd = (int) (T < 8 ? T : 8);
                    int32_t host_tok[8] = {0}, fg[12] = {0}, kg[12] = {0};
                    for (int i = 0; i < nd; ++i) host_tok[i] = h_tok_[i];
                    const bool ok1 = cudaMemcpy(fg, g_stage_dev + (size_t) 15 * 8192, sizeof(fg),
                                                cudaMemcpyDeviceToHost) == cudaSuccess;
                    const bool ok2 = cudaMemcpy(kg, g_stage_dev + (size_t) 15 * 8192 + 4096, sizeof(kg),
                                                cudaMemcpyDeviceToHost) == cudaSuccess;
                    std::fprintf(stderr, "mtp tokdump: cell=%lld a=%d T=%d host=", (long long) p, a, (int) T);
                    for (int i = 0; i < nd; ++i) std::fprintf(stderr, " %d", (int) host_tok[i]);
                    std::fprintf(stderr, " | full=");
                    for (int i = 0; i < nd; ++i) std::fprintf(stderr, " %d", (int) fg[i]);
                    std::fprintf(stderr, " | kv=");
                    for (int i = 0; i < (int) T && i < 8; ++i) std::fprintf(stderr, " %d", (int) kg[i]);
                    std::fprintf(stderr, " | sel full=%d,%d kv=%d,%d | copies %d/%d\n",
                                 (int) fg[8], (int) fg[9], (int) kg[8], (int) kg[9], ok1 ? 1 : 0, ok2 ? 1 : 0);
                    std::FILE* tf = std::fopen(tokdump, "ab");
                    if (tf != nullptr) {
                        const int32_t hdr[4] = {(int32_t) p, (int32_t) a, (int32_t) T, (int32_t) nd};
                        std::fwrite(hdr, sizeof(hdr), 1, tf);
                        std::fwrite(host_tok, sizeof(int32_t), 8, tf);
                        std::fwrite(fg, sizeof(int32_t), 12, tf);
                        std::fwrite(kg, sizeof(int32_t), 12, tf);
                        std::fclose(tf);
                    }
                    // ADAPTED (2026-10-10, THIS PORT): the K/V-only pass's own vcur_/kcur_ (before the cache), in a
                    // companion file `<path>.kvpre` - 512 floats of V then 512 of K, so the prefill row can be
                    // followed past `mixed_` (see the note in record_forward).
                    const int64_t kvf = (int64_t) g_->n_head_kv * (int64_t) g_->head_dim;
                    std::vector<float> pre((size_t) kvf * 2);
                    if (cudaMemcpy(pre.data(), g_stage_dev + (size_t) 15 * 8192 + 4200, (size_t) kvf * 4,
                                   cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(pre.data() + kvf, g_stage_dev + (size_t) 15 * 8192 + 4720, (size_t) kvf * 4,
                                   cudaMemcpyDeviceToHost) == cudaSuccess) {
                        std::FILE* pf2 = std::fopen((std::string(tokdump) + ".kvpre").c_str(), "ab");
                        if (pf2 != nullptr) {
                            const int32_t h2[4] = {(int32_t) p, (int32_t) a, (int32_t) T, (int32_t) kvf};
                            std::fwrite(h2, sizeof(h2), 1, pf2);
                            std::fwrite(pre.data(), 4, pre.size(), pf2);
                            std::fclose(pf2);
                        }
                    }
                    std::fflush(stderr);
                }
                // ADAPTED (2026-10-09, THIS PORT): THE WINDOW IDENTITY LIST, and the step/position records that go
                // with it.  The attention reads cells through `ident_`, whose length is `cap_` = the rounded window
                // - 32768 entries per query in the delivery configuration, on a prompt of 16 cells.  Which of those
                // entries are real cells, which are masked, and whether the list even ends at the query's own cell
                // is exactly what the full-window reproduction failure (cos 0.24) leaves open, and nothing else in
                // the block can answer it.  STRATA_MTP_DUMP_WINDOW=<path>:
                //   int32 {cell, a, cap_, step[4], pos[0]} then cap_ int32 identity entries.
                if (const char* wdump = std::getenv("STRATA_MTP_DUMP_WINDOW"); wdump != nullptr && ident_ != nullptr && cap_ > 0) {
                    std::vector<int32_t> id((size_t) cap_);
                    int32_t st4[4] = {0}, ps0 = 0;
                    cudaMemcpy(st4, step_ + (2 * max_t_ - 1) * 4, sizeof(st4), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&ps0, pos_ + (2 * max_t_ - 1) * g_->n_head, sizeof(ps0), cudaMemcpyDeviceToHost);
                    if (cudaMemcpy(id.data(), ident_, id.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess) {
                        std::FILE* wf = std::fopen(wdump, "ab");
                        if (wf != nullptr) {
                            const int32_t hdr3[7] = {(int32_t) p, (int32_t) a, (int32_t) cap_,
                                                     st4[0], st4[1], st4[2], st4[3]};
                            std::fwrite(hdr3, sizeof(hdr3), 1, wf);
                            std::fwrite(&ps0, sizeof(ps0), 1, wf);
                            std::fwrite(id.data(), sizeof(int32_t), id.size(), wf);
                            std::fclose(wf);
                            int32_t lo = id[0], hi = id[0], inrange = 0, neg = 0;
                            for (int32_t v : id) { lo = std::min(lo, v); hi = std::max(hi, v); if (v < 0) ++neg; if (v >= 0 && v < 64) ++inrange; }
                            std::fprintf(stderr, "mtp windowdump: cell=%lld cap=%lld step=%d,%d,%d,%d pos0=%d  id[0]=%d id[1]=%d id[last]=%d  min=%d max=%d neg=%d in[0,64)=%d\n",
                                         (long long) p, (long long) cap_, st4[0], st4[1], st4[2], st4[3], ps0,
                                         id[0], id.size() > 1 ? id[1] : -1, id[id.size() - 1], lo, hi, neg, inrange);
                        }
                    }
                    std::fflush(stderr);
                }
                // inside the MoE: the router's logits and the expert ids it picked (out of range ids would make
                // the blob read land anywhere), the combine weights, the grouped parts, the shared expert.
                {
                    float lg[2] = {0}, w2[2] = {0}, sh0 = 0.0f, ps[8] = {0};
                    uint16_t xb0 = 0;
                    int32_t ids[8] = {0};
                    const int kk = (int) ss_->k;
                    const size_t stride = (size_t) g_->n_embd;
                    cudaMemcpy(lg, logits_, sizeof(lg), cudaMemcpyDeviceToHost);
                    cudaMemcpy(ids, ids_, sizeof(ids), cudaMemcpyDeviceToHost);
                    cudaMemcpy(w2, w_, sizeof(w2), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&sh0, shared_, sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&xb0, x_bf16_, sizeof(uint16_t), cudaMemcpyDeviceToHost);
                    for (int k = 0; k < kk && k < 8; ++k)
                        cudaMemcpy(&ps[k], parts_ + (size_t) k * stride, sizeof(float), cudaMemcpyDeviceToHost);
                    const uint32_t xbu = ((uint32_t) xb0) << 16;
                    float xbf = 0.0f;
                    std::memcpy(&xbf, &xbu, sizeof(float));
                    std::fprintf(stderr, "mtp debug moe: ne=%d K=%d gate=%g %g ids=%d %d %d %d %d %d %d %d w=%g %g\n",
                                 ne_, kk, lg[0], lg[1], ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
                                 w2[0], w2[1]);
                    std::fprintf(stderr, "               shared[0]=%g xbf16_in[0]=%g parts[k*N]=%g %g %g %g %g %g %g %g\n",
                                 sh0, xbf, ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], ps[6], ps[7]);
                    // the other half of the expert dot: the quantized activation and its scales
                    {
                        float xs[4] = {0};
                        uint8_t xq[8] = {0};
                        cudaMemcpy(xs, hit_xs_, sizeof(xs), cudaMemcpyDeviceToHost);
                        cudaMemcpy(xq, hit_xq_, sizeof(xq), cudaMemcpyDeviceToHost);
                        std::fprintf(stderr, "               hit_xs=%g %g %g %g  hit_xq=%u %u %u %u %u %u %u %u\n",
                                     xs[0], xs[1], xs[2], xs[3], xq[0], xq[1], xq[2], xq[3], xq[4], xq[5], xq[6], xq[7]);
                        // did the grouping even happen?  grp_ptr_[i] must equal experts_ + ids_[i] * BLOB
                        int32_t gc = 0, gs[4] = {0};
                        unsigned long long gp = 0;
                        cudaMemcpy(&gc, grp_counts_, sizeof(gc), cudaMemcpyDeviceToHost);
                        cudaMemcpy(gs, grp_start_, sizeof(gs), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&gp, grp_ptr_, sizeof(gp), cudaMemcpyDeviceToHost);
                        std::fprintf(stderr, "               grp_counts=%d grp_start=%d %d %d %d  grp_ptr[0]-experts=%lld (ids[0]*BLOB=%lld)\n",
                                     gc, gs[0], gs[1], gs[2], gs[3], (long long) gp - (long long) (uintptr_t) experts_,
                                     (long long) ids[0] * (long long) (884736));
                        std::fflush(stderr);
                    }
                }
            }
        }
    }
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
}

}  // namespace strata::core
