#!/usr/bin/env python3
"""Build the id -> bytes table used to detokenize engine output.

The engine emits raw token ids (it carries no tokenizer).  This script turns a
byte-level BPE vocab.json into a flat table of

    <id>\t<hex bytes>

so that a caller can concatenate the pieces for a sequence of ids and decode the
result as UTF-8.  Concatenating per-token *bytes* (rather than per-token strings)
is what makes multi-byte characters split across tokens come out right.

Usage:
    python make_vocab_bytes.py <vocab.json> <out.tsv>

vocab.json comes from the model's tokenizer (the pack directory copies it under
tokenizer/vocab.json when the pack is built).
"""
import json
import sys


def bytes_to_unicode():
    """The GPT-2 / Qwen byte-level BPE alphabet -> printable-character map."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: make_vocab_bytes.py <vocab.json> <out.tsv>")

    with open(sys.argv[1], encoding="utf-8") as fh:
        vocab = json.load(fh)

    char_to_byte = {ch: b for b, ch in bytes_to_unicode().items()}

    rows = []
    for token, tid in vocab.items():
        buf = bytearray()
        for ch in token:
            buf.append(char_to_byte.get(ch, ord(ch) & 0xFF))
        rows.append((int(tid), bytes(buf)))
    rows.sort()

    with open(sys.argv[2], "w", encoding="ascii", newline="\n") as fh:
        fh.write("# id<TAB>hexbytes -- byte-level BPE pieces, pre-decoded\n")
        for tid, piece in rows:
            fh.write("%d\t%s\n" % (tid, piece.hex()))

    print("entries %d, largest piece %d bytes -> %s"
          % (len(rows), max(len(p) for _, p in rows), sys.argv[2]))


if __name__ == "__main__":
    main()