"""Chat over the transcripts: per-note chunk index, retrieval, prompt with [alias#turn] refs, two model backends, sessions on disk.

Nothing here touches the GPU: embeddings come from the local Ollama (bge-m3), answers from the pipeline's Ollama or from
`claude -p` on this machine. Every function takes the requester's root, so a guest's notes never mix with the owner's.
"""
import hashlib, json, os, re, shutil, subprocess, time, uuid
from pathlib import Path

import numpy as np

import pipeline as pl

EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHUNK_CHARS = 600
CONTEXT_CHARS = 17000
HISTORY_CHARS = 8000
TOP_K = 24
PAD = 2
NOTE_ID = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{6}-")  # note ids are dated; anything else in a root (chats/, speakers.json) is not a note


# ---------- index ----------
def turn_text(seg):
    return seg.get("hinglish") or seg.get("text") or ""


def chunk_turns(segments, size=CHUNK_CHARS):
    """Consecutive turns grouped until `size` chars, one turn of overlap between neighbours. No speaker names in the text."""
    chunks, i = [], 0
    while i < len(segments):
        j, parts = i, []
        while j < len(segments) and (not parts or sum(len(p) + 1 for p in parts) < size):
            parts.append(turn_text(segments[j])); j += 1
        last = j - 1
        chunks.append({"first": i, "last": last, "start": segments[i]["start"], "end": segments[last]["end"],
                       "speakers": sorted({segments[k]["speaker"] for k in range(i, last + 1)}), "text": "\n".join(parts)})
        if last >= len(segments) - 1:
            break
        i = last  # overlap by one turn
    return chunks


def embed(texts):
    import urllib.request
    rows = []
    for k in range(0, len(texts), 32):
        req = urllib.request.Request(EMBED_URL + "/api/embed", data=json.dumps({"model": EMBED_MODEL, "input": texts[k:k + 32]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            rows += json.loads(r.read())["embeddings"]
    m = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def index_note(note_dir, embed=embed):
    """Writes chunks.json + chunks.npy next to transcript.json. Skips the embedding call when the text is unchanged."""
    note_dir = Path(note_dir)
    t = pl.read_json(note_dir / "transcript.json", None)
    if t is None:
        return 0
    chunks = chunk_turns(t.get("segments", []))
    h = hashlib.sha1("\n\n".join(c["text"] for c in chunks).encode()).hexdigest()
    old = pl.read_json(note_dir / "chunks.json", {})
    if old.get("hash") == h and (note_dir / "chunks.npy").exists():
        return len(old["chunks"])
    m = embed([c["text"] for c in chunks]) if chunks else np.zeros((0, 1024), dtype=np.float32)
    np.save(note_dir / "chunks.npy", m)
    pl.write_json(note_dir / "chunks.json", {"hash": h, "chunks": chunks})
    return len(chunks)


def load_index(root, note=None):
    out = []
    for d in sorted(root.iterdir()) if note is None else [root / note]:
        if d.is_dir() and NOTE_ID.match(d.name) and (d / "chunks.npy").exists():
            m = np.load(d / "chunks.npy")
            out.append((d.name, pl.read_json(d / "chunks.json", {}).get("chunks", []), m.reshape(-1, 1024) if m.size else np.zeros((0, 1024), dtype=np.float32)))
    return out
