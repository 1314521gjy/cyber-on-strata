"""tools/make_profile.py - the expert-cache profile (`data/expert-profile.bin`): every (layer, expert) pair, ranked.

`--expert-cache auto` fills the VRAM slots it can afford with the profile's pairs in order, so the ranking decides
which experts start resident (the adaptive tier then swaps in what a conversation routes most).  A profile that
ranks fewer pairs than a card can hold caps the cache (issue #46: a 32 GB card stopped at 8,000 slots); this tool
writes one that ranks all 24,576.

The order: the base profile's ranking (default: the shipped data/expert-profile.bin), then the pairs your routing
traces used, most frequent first, then every pair still missing, interleaved across the layers.  The base is only
*appended to*, never reordered - and the shipped profile already ranks all 24,576 pairs, so with it a trace is a
no-op.  `--reorder` ranks the traces first and lets the base fill the rest, which is how a workload's trace decides
the top; `--no-base` drops the base entirely.

    python tools/make_profile.py [TRACE ...] [--base data/expert-profile.bin | --no-base] [--reorder] [--out PATH]
                                 [--n-layer 40] [--n-expert 256]
                                 (`--n-layer` MUST equal the engine's n_layers; `--n-expert` covers a pruned
                                  model, e.g. GSQ-RCO Coder keeps 256 of 512)

A routing trace comes from a one-shot engine run with `--dump-routing FILE` (a prompt typical of your use; the
routed experts of every layer and position are written).  Point the model config's `--expert-profile` at the result.
"""
import argparse
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The canonical artifact's shape (Qwen3.8-Flash-Next: 48 layers x 512 experts).  BOTH are overridable on the
# command line: `--n-layer` is the half that used to be missing, and it is not cosmetic.  Without it,
# `--n-expert 256` on its own wrote a **48x256** table for a target with fewer layers - and the round-trip
# assert in `main()` CANNOT catch that, because `write_profile` and `read_profile` compare against the same
# module constant, so the wrong file passes its own check.  (`read_expert_profile` in the engine does compare
# the header against the model's geometry, so the mistake is caught at LOAD - loudly, but a full pack later.)
N_LAYER, N_EXPERT = 48, 512
MAGIC, VERSION = b"STRP", 1


def read_profile(path, n_layer=N_LAYER, n_expert=N_EXPERT):
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if (nl, ne) != (n_layer, n_expert):
        raise SystemExit(f"{path}: {nl}x{ne}, not {n_layer}x{n_expert}")
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def read_trace(path, n_layer=N_LAYER, n_expert=N_EXPERT):
    """(layer, k, k expert ids, k weights) records, as `--dump-routing` writes them."""
    blob = Path(path).read_bytes()
    off, freq = 0, defaultdict(int)
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        for e in struct.unpack_from("<%di" % k, blob, off):
            if 0 <= layer < n_layer and 0 <= e < n_expert:
                freq[(layer, e)] += 1
        off += 8 * k                                    # the ids and the weights
    return freq


def write_profile(path, ranked, n_layer=N_LAYER, n_expert=N_EXPERT):
    table = [[-1] * n_expert for _ in range(n_layer)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_layer, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_layer):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def rank_profile(base_pairs, trace_freq, n_expert=N_EXPERT, n_layer=N_LAYER, no_base=False, reorder=False):
    """The ranked (layer, expert) list and the counts it was built from (`base`, `trace`, `fill`).

    Default: the base's order, then the traces' pairs most frequent first, then the fill.  With `reorder` (or
    `no_base`) the traces' pairs come first, so a base that already ranks every pair no longer hides them; the base
    (unless `no_base`) and then the fill follow.  A base is never reordered against itself - it only ever fills."""
    ranked, seen, counts = [], set(), defaultdict(int)

    def take(pairs, key):
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)
                counts[key] += 1

    trace_pairs = [p for p, _ in sorted(trace_freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    if no_base or reorder:
        take(trace_pairs, "trace")
    if not no_base:
        take(base_pairs, "base")
    if not (no_base or reorder):
        take(trace_pairs, "trace")
    take(((layer, e) for e in range(n_expert) for layer in range(n_layer)), "fill")
    return ranked, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="routing traces from --dump-routing")
    ap.add_argument("--base", default=str(ROOT / "data" / "expert-profile.bin"), help="ranking to keep first")
    ap.add_argument("--no-base", action="store_true", help="rank by the traces only")
    ap.add_argument("--reorder", action="store_true",
                    help="rank the traces' pairs before the base's (the base, then the fill, follow); needed with a "
                         "base that already ranks every pair, where --base alone cannot change the order")
    ap.add_argument("--out", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--n-layer", type=int, default=N_LAYER,
                    help="layers in the profile (default 48).  MUST equal the engine's ModelGeometry::n_layers: "
                         "read_expert_profile compares the header against the model and refuses a mismatch, so 48 "
                         "here for a 40-layer target is a load-time error - loud, but a whole pack late")
    ap.add_argument("--n-expert", type=int, default=N_EXPERT, help="experts per layer (default 512)")
    a = ap.parse_args()

    nl, ne = a.n_layer, a.n_expert
    freq = defaultdict(int)
    for t in a.traces:
        for p, c in read_trace(t, n_layer=nl, n_expert=ne).items():
            freq[p] += c
    base_pairs = [] if a.no_base else read_profile(a.base, n_layer=nl, n_expert=ne)
    ranked, counts = rank_profile(base_pairs, freq, n_expert=ne, n_layer=nl, no_base=a.no_base, reorder=a.reorder)
    write_profile(a.out, ranked, n_layer=nl, n_expert=ne)
    assert read_profile(a.out, n_layer=nl, n_expert=ne) == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {len(ranked)} ranked pairs ({counts['base']} from the base, {counts['trace']} from the "
          f"traces, {counts['fill']} filled in)")


if __name__ == "__main__":
    main()
