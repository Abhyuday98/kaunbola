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


# ---------- retrieval ----------
KEY_TOKEN = re.compile(r"\b(?:[A-Z][a-z]+|\d[\d:/.-]*\d|\d)\b")
FOLLOW = re.compile(r"^(and|or|but|what about|aur|phir|uske|uska|usne|wo|woh|she|he|they|it|that|then|also)\b", re.I)


def note_json(root, note):
    return pl.read_json(root / note / "transcript.json", None)


def grep_leg(root, query, note):
    """(note_id, turn) for every turn containing an exact name/number token of the query; a speaker's name hits all their turns."""
    toks = {t.lower() for t in KEY_TOKEN.findall(query)}
    hits = set()
    if not toks:
        return hits
    for d in ([root / note] if note else sorted(root.iterdir())):
        if not (d.is_dir() and NOTE_ID.match(d.name)):
            continue
        t = note_json(root, d.name)
        if not t:
            continue
        names = {(v or "").lower(): k for k, v in (t.get("speakers") or {}).items() if v}
        for i, s in enumerate(t.get("segments", [])):
            hay = " ".join((s.get("hinglish") or "", s.get("text") or "", s.get("english") or "")).lower()
            if any(tok in hay or names.get(tok) == s["speaker"] for tok in toks):
                hits.add((d.name, i))
    return hits


def query_text(question, prev_question):
    q = question.strip()
    if prev_question and (len(q.split()) < 6 or FOLLOW.match(q)):
        return prev_question.strip() + " " + q
    return q


def _merge(ranges):
    """Merge overlapping or touching ranges per note, keeping insertion order of the first range of each group."""
    out = []
    for r in ranges:
        for o in out:
            if o["note"] == r["note"] and r["first"] <= o["last"] + 1 and r["last"] >= o["first"] - 1:
                o["first"], o["last"], o["score"] = min(o["first"], r["first"]), max(o["last"], r["last"]), max(o["score"], r["score"])
                break
        else:
            out.append(dict(r))
    return out


def retrieve(root, query, note=None, embed=embed, top_k=TOP_K):
    index = load_index(root, note)
    scored = []
    if index and any(m.shape[0] for _, _, m in index):
        q = embed([query])[0]
        for nid, chunks, m in index:
            if m.shape[0]:
                for i, sc in zip(range(m.shape[0]), m @ q):
                    scored.append((float(sc), nid, chunks[i]["first"], chunks[i]["last"]))
    scored.sort(reverse=True)
    ranges = [{"note": nid, "first": f, "last": l, "score": sc} for sc, nid, f, l in scored[:top_k]]
    ranges += [{"note": nid, "first": t, "last": t, "score": 0.5} for nid, t in grep_leg(root, query, note)]
    counts = {}
    for r in ranges:
        counts.setdefault(r["note"], len((note_json(root, r["note"]) or {}).get("segments", [])))
        r["first"], r["last"] = max(0, r["first"] - PAD), min(counts[r["note"]] - 1, r["last"] + PAD)
    return sorted(_merge(ranges), key=lambda r: (r["note"], r["first"]))


def range_chars(root, ranges):
    total = 0
    for r in ranges:
        segs = (note_json(root, r["note"]) or {}).get("segments", [])
        total += sum(len(turn_text(s)) + 30 for s in segs[r["first"]:r["last"] + 1])
    return total


def fits_whole(root, note):
    n = len((note_json(root, note) or {}).get("segments", []))
    return n > 0 and range_chars(root, [{"note": note, "first": 0, "last": n - 1}]) <= CONTEXT_CHARS


def union_context(context, new, root):
    """Session context = everything retrieved so far, in first-seen order. Over the cap: drop oldest ranges; a lone oversized range is cut at its tail."""
    ctx = _merge(list(context) + list(new))
    while len(ctx) > 1 and range_chars(root, ctx) > CONTEXT_CHARS:
        ctx.pop(0)
    while ctx and range_chars(root, ctx) > CONTEXT_CHARS and ctx[0]["last"] > ctx[0]["first"]:
        ctx[0]["last"] -= max(1, (ctx[0]["last"] - ctx[0]["first"]) // 10)
    return ctx
