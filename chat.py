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
    """(note_id, turn) for every turn containing a rare exact token of the query (a name, a number); a speaker's name hits all their turns.
    A token found in more than 2% of all turns (sentence starters like "What", "Kya") carries no information and is ignored."""
    toks = {t.lower() for t in KEY_TOKEN.findall(query)}
    if not toks:
        return set()
    per_tok, total = {t: set() for t in toks}, 0
    for d in ([root / note] if note else sorted(root.iterdir())):
        if not (d.is_dir() and NOTE_ID.match(d.name)):
            continue
        t = note_json(root, d.name)
        if not t:
            continue
        names = {(v or "").lower(): k for k, v in (t.get("speakers") or {}).items() if v}
        for i, s in enumerate(t.get("segments", [])):
            total += 1
            hay = " ".join((s.get("hinglish") or "", s.get("text") or "", s.get("english") or "")).lower()
            for tok in toks:
                if names.get(tok) == s["speaker"]:
                    per_tok[tok].add((d.name, i, True))
                elif tok in hay:
                    per_tok[tok].add((d.name, i, False))
    hits = set()
    for tok, found in per_tok.items():
        if sum(1 for h in found if not h[2]) <= max(20, total * 0.02):  # text hits of a common word are noise; speaker-name hits always count
            hits |= {(n, i) for n, i, _ in found}
        else:
            hits |= {(n, i) for n, i, by_name in found if by_name}
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
    merged = sorted(_merge(ranges), key=lambda r: -r["score"])
    while merged and range_chars(root, merged) > CONTEXT_CHARS:  # over the cap: lowest scores go first
        merged.pop()
    return sorted(merged, key=lambda r: (r["note"], r["first"]))


def range_chars(root, ranges):
    return sum(len(render_range(root, r["note"], "n1", r["first"], r["last"])) + 1 for r in ranges)


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


# ---------- prompt ----------
ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"},
                 "sources": {"type": "array", "items": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}}},
                 "required": ["answer", "sources"]}
SYSTEM_PROMPT = ("You answer questions about recorded conversations. You are given transcript turns, each starting with a reference like [n1#42], "
                 "under headers naming the recording, its date and its summary. Answer fully, in the language the question is asked in (Hinglish questions "
                 "get Hinglish answers). Use only the turns given: never add facts from elsewhere, never guess. When you attribute a statement to a person, "
                 "quote their words. Put the reference of every turn you rely on right after the sentence that uses it, like this: Priya said the draft will "
                 "be ready by Monday [n1#42]. If the turns do not contain the answer, say so plainly and do not invent one. Return JSON with \"answer\" "
                 "(markdown; paragraphs and bullet lists only) and \"sources\" (every reference you used, in order of first use).")
REF = re.compile(r"\[(n\d+#\d+)\]")


def aliases_for(session, note_ids):
    al = session.setdefault("aliases", {})
    for nid in note_ids:
        if nid not in al.values():
            al[f"n{len(al) + 1}"] = nid
    return al


def spk_name(t, key):
    return (t.get("speakers") or {}).get(key) or pl.spk_label(key)


def render_range(root, note_id, alias, first, last):
    t = note_json(root, note_id) or {}
    segs = t.get("segments", [])
    return "\n".join(f"[{alias}#{i}] {pl.fmt_ts(segs[i]['start'])} {spk_name(t, segs[i]['speaker'])}: {turn_text(segs[i])}" for i in range(first, min(last, len(segs) - 1) + 1))


def render_context(root, ranges, aliases):
    by_alias = {v: k for k, v in aliases.items()}
    parts, refs = [], set()
    for alias, nid in aliases.items():
        rs = [r for r in ranges if r["note"] == nid]
        if not rs:
            continue
        t = note_json(root, nid) or {}
        parts.append(f"## {alias}: {t.get('title', nid)} ({(t.get('created') or '')[:10]})\nSummary: {(t.get('summary') or '').strip()}")
        for r in sorted(rs, key=lambda r: r["first"]):
            block = render_range(root, nid, alias, r["first"], r["last"])
            parts.append(block); refs |= set(REF.findall(block))
    return "\n\n".join(parts), refs


def notes_list(root):
    rows = []
    for d in sorted(root.iterdir(), reverse=True):
        if d.is_dir() and NOTE_ID.match(d.name):
            t = note_json(root, d.name)
            if t and t.get("segments"):
                rows.append(f"- {t.get('title', d.name)} ({(t.get('created') or '')[:10]}, {round((t.get('duration') or 0) / 60)} min)")
    return "Recordings available:\n" + "\n".join(rows)


def prompt_chars(messages):
    return sum(len(m["content"]) for m in messages)


def build_messages(session, question, context_text, listing):
    limit = int(pl.NUM_CTX * pl.CHARS_PER_TOKEN * 0.85)
    hist = [{"role": m["role"], "content": m.get("text", "")} for m in session.get("messages", []) if not m.get("pending") and m.get("text")]
    while hist and sum(len(m["content"]) for m in hist) > HISTORY_CHARS:
        hist.pop(0)
    head = ((listing + "\n\n") if listing else "") + "Context:\n" + context_text
    msgs = lambda h: [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": h}, *hist, {"role": "user", "content": question}]
    while hist and prompt_chars(msgs(head)) > limit:
        hist.pop(0)
    if prompt_chars(msgs(head)) > limit:
        head = head[:limit - prompt_chars(msgs(""))].rsplit("\n", 1)[0]
    return msgs(head)


def validate(answer, refs):
    """Keep only refs that were in the context; renumber markers in the text to [k]; drop markers of unknown refs."""
    kept = []
    for s in answer.get("sources") or []:
        r = s.get("ref") if isinstance(s, dict) else s
        if r in refs and r not in kept:
            kept.append(r)
    text = str(answer.get("answer") or "")
    for r in REF.findall(text):
        if r in refs and r not in kept:
            kept.append(r)
    def sub(m):
        r = m.group(1)
        return f"[{kept.index(r) + 1}]" if r in kept else ""
    text = re.sub(r" ?\[(n\d+#\d+)\]", lambda m: (" " + sub(m)) if sub(m) else "", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip(), kept


def resolve(root, ref, aliases):
    alias, _, turn = ref.partition("#")
    nid = aliases.get(alias)
    t = note_json(root, nid) if nid else None
    if not t or not turn.isdigit() or int(turn) >= len(t.get("segments", [])):
        return None
    s = t["segments"][int(turn)]
    return {"ref": ref, "note": nid, "title": t.get("title", nid), "created": t.get("created", ""), "turn": int(turn), "start": s["start"], "end": s["end"],
            "speaker": spk_name(t, s["speaker"]), "hinglish": turn_text(s), "english": s.get("english") or ""}


# ---------- models ----------
CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or str(Path.home() / ".npm-global/bin/claude")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "")
CLAUDE_CWD = Path("~/.config/audio-notes/claude-cwd").expanduser()  # empty: no CLAUDE.md, no project memory, no skills get loaded


def call_local(messages):
    return pl.llm_messages(messages, ANSWER_SCHEMA, num_predict=2048)


def claude_argv(prompt):
    # the account's MCP connectors alone are ~140k tokens of system prompt; these flags bring a call down to ~4k
    return [CLAUDE_BIN, "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(ANSWER_SCHEMA), "--append-system-prompt", SYSTEM_PROMPT,
            "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--tools", "", "--no-session-persistence",
            *(["--model", CLAUDE_MODEL] if CLAUDE_MODEL else [])]


def flatten(messages):
    return "\n\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}:\n{m['content']}" for m in messages if m["role"] != "system")


def call_claude(messages):
    CLAUDE_CWD.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(claude_argv(flatten(messages)), cwd=CLAUDE_CWD, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:300] or "claude failed")
    out = json.loads(r.stdout)
    if isinstance(out.get("structured_output"), dict):
        return out["structured_output"]
    res = out.get("result")
    return res if isinstance(res, dict) else json.loads(res)


MODELS = {"local": call_local, "claude": call_claude}


# ---------- sessions ----------
SID = re.compile(r"^[0-9A-Za-z-]+$")


def chats_dir(root):
    d = root / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def now():
    return time.time()


def save_session(root, s):
    s["updated"] = now()
    pl.write_json(chats_dir(root) / f"{s['id']}.json", s)
    return s


def new_session(root, note, model):
    s = {"id": time.strftime("%Y-%m-%d-%H%M%S-") + uuid.uuid4().hex[:4], "created": now(), "updated": now(), "title": "New chat",
         "note": note, "model": model, "aliases": {}, "context": [], "messages": []}
    return save_session(root, s)


def load_session(root, sid):
    if not SID.match(sid or ""):
        raise KeyError(sid)
    s = pl.read_json(chats_dir(root) / f"{sid}.json", None)
    if s is None:
        raise KeyError(sid)
    return s


def list_sessions(root, note=None):
    out = []
    for p in chats_dir(root).glob("*.json"):
        s = pl.read_json(p, None)
        if s and (note is None or s.get("note") == note):
            out.append({"id": s["id"], "title": s["title"], "note": s.get("note"), "model": s.get("model", "local"), "updated": s["updated"], "count": len(s["messages"])})
    return sorted(out, key=lambda s: s["updated"], reverse=True)


def delete_session(root, sid):
    load_session(root, sid)
    (chats_dir(root) / f"{sid}.json").unlink()


def clear_session(root, sid):
    s = load_session(root, sid)
    s["messages"], s["context"] = [], []
    return save_session(root, s)


def pending(s):
    return bool(s["messages"]) and s["messages"][-1]["role"] == "assistant" and s["messages"][-1].get("pending", False)


def queue_question(root, sid, text, model):
    s = load_session(root, sid)
    if pending(s):
        raise ValueError("busy")
    if not any(m["role"] == "user" for m in s["messages"]):
        s["title"] = text.strip()[:60]
    s["model"] = model
    s["messages"] += [{"role": "user", "text": text.strip(), "ts": now()},
                      {"role": "assistant", "text": "", "sources": [], "model": model, "ts": now(), "pending": True, "detail": "retrieving"}]
    return save_session(root, s)


def answer(root, sid, embed=embed, models=MODELS):
    """Runs the pending question of a session to completion (or to an error). Safe to call from a thread."""
    s = load_session(root, sid)
    if not pending(s):
        return s
    m = s["messages"][-1]
    try:
        users = [x["text"] for x in s["messages"] if x["role"] == "user"]
        question, prev = users[-1], (users[-2] if len(users) > 1 else None)
        note = s.get("note")
        if note and fits_whole(root, note):
            n = len((note_json(root, note) or {}).get("segments", []))
            new = [{"note": note, "first": 0, "last": n - 1, "score": 1.0}]
        else:
            new = retrieve(root, query_text(question, prev), note, embed=embed)
        s["context"] = union_context(s.get("context", []), new, root)
        aliases = aliases_for(s, [r["note"] for r in s["context"]])
        text, refs = render_context(root, s["context"], aliases)
        msgs = build_messages({**s, "messages": s["messages"][:-2]}, question, text, None if note else notes_list(root))
        m["detail"] = "thinking"; save_session(root, s)
        call = models[m.get("model") or "local"]
        if m.get("model", "local") == "local" and pl.LLM_SHARES_GPU:
            m["detail"] = "waiting for the GPU"; save_session(root, s)
            with pl.MODELS.lock:
                pl.MODELS.unload()
                out = call(msgs)
        else:
            out = call(msgs)
        m["text"], kept = validate(out, refs)
        m["sources"] = [src for src in (resolve(root, r, aliases) for r in kept) if src]
    except Exception as e:
        m["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    m["pending"] = False; m.pop("detail", None)
    return save_session(root, s)


def mark_interrupted(root):
    for p in chats_dir(root).glob("*.json"):
        s = pl.read_json(p, None)
        if s and pending(s):
            s["messages"][-1].update({"pending": False, "error": "interrupted by a restart"}); s["messages"][-1].pop("detail", None)
            save_session(root, s)


def index_missing(root, embed=embed):
    built = 0
    for d in sorted(root.iterdir()):
        if d.is_dir() and NOTE_ID.match(d.name) and (d / "transcript.json").exists():
            had = (d / "chunks.npy").exists() and pl.read_json(d / "chunks.json", {}).get("hash")
            try:
                index_note(d, embed=embed)
                built += 0 if had and pl.read_json(d / "chunks.json", {}).get("hash") == had else 1
            except Exception as e:
                print(f"index failed for {d.name}: {e}")
    return built
