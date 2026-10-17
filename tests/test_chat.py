import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def test_note_dirs_skips_chats():
    import server
    root = Path(tempfile.mkdtemp())
    (root / "2026-10-17-120000-a-note").mkdir(); (root / "chats").mkdir(); (root / "speakers.json").write_text("{}")
    assert [d.name for d in server.note_dirs(root)] == ["2026-10-17-120000-a-note"]


def fake_embed(texts):  # deterministic: bag of lowercase 3-grams hashed into 1024 dims
    out = np.zeros((len(texts), 1024), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t.lower()
        for j in range(len(t) - 2):
            out[i, hash(t[j:j + 3]) % 1024] += 1
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


SEGS = [{"start": i * 10.0, "end": i * 10.0 + 9, "speaker": f"SPEAKER_0{i % 2}", "text": f"t{i}",
         "hinglish": ("budget line pe baat hui " if i == 7 else f"turn {i} ") * 20, "english": f"e{i}"} for i in range(30)]


def make_note(root, nid, segs, title="A note"):
    d = root / nid; d.mkdir(parents=True)
    json.dump({"title": title, "created": "2026-10-09T10:00:00", "sha256": "x"}, open(d / "meta.json", "w"))
    json.dump({"id": nid, "title": title, "created": "2026-10-09T10:00:00", "duration": 300, "speakers": {"SPEAKER_00": "Priya", "SPEAKER_01": None},
               "summary": "About the budget.", "segments": segs}, open(d / "transcript.json", "w"))
    json.dump({"state": "done"}, open(d / "status.json", "w"))
    return d


def test_chunk_turns_overlap_and_size():
    import chat
    chunks = chat.chunk_turns(SEGS, size=600)
    assert chunks[0]["first"] == 0 and chunks[1]["first"] == chunks[0]["last"]   # one turn overlap
    assert all(len(c["text"]) >= 600 or c is chunks[-1] for c in chunks)
    assert chunks[0]["speakers"] == ["SPEAKER_00", "SPEAKER_01"] and "SPEAKER" not in chunks[0]["text"]
    assert chat.chunk_turns([SEGS[0]])[0]["last"] == 0 and chat.chunk_turns([]) == []


def test_chunk_turns_terminates_on_long_turn():
    import chat, threading
    segs = [dict(SEGS[0], hinglish="x " * 700), dict(SEGS[1], hinglish="short"), dict(SEGS[2], hinglish="y " * 700), dict(SEGS[3], hinglish="end")]
    out = []
    th = threading.Thread(target=lambda: out.append(chat.chunk_turns(segs, size=600)), daemon=True); th.start(); th.join(2)
    assert not th.is_alive(), "chunk_turns hangs on a single turn longer than the chunk size"
    firsts = [c["first"] for c in out[0]]
    assert firsts == sorted(set(firsts)) and out[0][-1]["last"] == 3


def test_index_note_roundtrip_and_skip():
    import chat
    root = Path(tempfile.mkdtemp()); d = make_note(root, "2026-10-09-100000-one", SEGS)
    n = chat.index_note(d, embed=fake_embed)
    assert n == len(chat.chunk_turns(SEGS)) and (d / "chunks.npy").exists()
    calls = []
    assert chat.index_note(d, embed=lambda t: calls.append(t) or fake_embed(t)) == n and calls == []  # hash unchanged: no embedding
    [(nid, chunks, m)] = chat.load_index(root)
    assert nid == d.name and m.shape == (n, 1024) and abs(float(np.linalg.norm(m[0])) - 1) < 1e-5
    empty = make_note(root, "2026-10-09-100001-empty", [])
    assert chat.index_note(empty, embed=fake_embed) == 0 and chat.load_index(root, empty.name)[0][2].shape[0] == 0


def test_retrieve_cross_script_and_grep():
    import chat
    root = Path(tempfile.mkdtemp()); d = make_note(root, "2026-10-09-100000-one", SEGS); chat.index_note(d, embed=fake_embed)
    r1 = chat.retrieve(root, "budget line pe kya baat hui", embed=fake_embed, top_k=2)
    assert any(x["first"] <= 7 <= x["last"] for x in r1), r1            # turn 7 is the budget turn
    assert all(x["first"] >= 0 and x["last"] < len(SEGS) for x in r1)
    hits = chat.grep_leg(root, "What did Priya say at 00:40", None)      # "Priya" names SPEAKER_00 -> every even turn; "What" and "00:40" match no text
    assert hits == {(d.name, t) for t in range(0, 30, 2)}


def test_grep_ignores_common_words_and_retrieve_caps_by_score():
    import chat
    root = Path(tempfile.mkdtemp())
    segs = [{**x, "hinglish": "What about " + x["hinglish"]} for x in SEGS * 40]   # "what" in every turn of a 1200-turn note
    d = make_note(root, "2026-10-09-100000-one", segs); chat.index_note(d, embed=fake_embed)
    hits = chat.grep_leg(root, "What did Priya say about the budget", None)
    assert hits and all(t % 2 == 0 for _, t in hits)                                 # "What" is everywhere: ignored; Priya's turns only
    r = chat.retrieve(root, "What did Priya say about the budget", embed=fake_embed, top_k=200)
    assert chat.range_chars(root, r) <= chat.CONTEXT_CHARS and r


def test_query_text_expands_followups():
    import chat
    assert chat.query_text("and what did she say after that?", "what did Priya decide about the budget").startswith("what did Priya decide")
    assert chat.query_text("Kal ki meeting mein Rohan ne scope ke baare mein kya bola?", "x") == "Kal ki meeting mein Rohan ne scope ke baare mein kya bola?"


def test_union_context_caps_from_front():
    import chat
    root = Path(tempfile.mkdtemp()); d = make_note(root, "2026-10-09-100000-one", SEGS * 40)  # 1200 turns
    ctx = chat.union_context([], [{"note": d.name, "first": 0, "last": 3, "score": 1}], root)
    ctx = chat.union_context(ctx, [{"note": d.name, "first": 2, "last": 5, "score": 1}, {"note": d.name, "first": 20, "last": 22, "score": 1}], root)
    assert [(r["first"], r["last"]) for r in ctx] == [(0, 5), (20, 22)]
    big = [{"note": d.name, "first": i, "last": i, "score": 1} for i in range(1200)]  # far over CONTEXT_CHARS once rendered
    ctx2 = chat.union_context(ctx, big, root)
    assert chat.range_chars(root, ctx2) <= chat.CONTEXT_CHARS and (ctx2[0]["first"], ctx2[0]["last"]) != (0, 5)


def test_render_validate_resolve():
    import chat
    root = Path(tempfile.mkdtemp()); d = make_note(root, "2026-10-09-100000-one", SEGS)
    s = {"aliases": {}, "messages": [], "context": []}
    aliases = chat.aliases_for(s, [d.name]); assert aliases == {"n1": d.name}
    text, refs = chat.render_context(root, [{"note": d.name, "first": 6, "last": 8, "score": 1}], aliases)
    assert "## n1: A note (2026-10-09)" in text and "[n1#7] 00:01:10 Priya:" not in text and "[n1#7] 00:01:10 Speaker 2:" in text and refs == {"n1#6", "n1#7", "n1#8"}
    ans, kept = chat.validate({"answer": "Budget was raised [n1#7] and ignored [n1#9]. Also [n1#6].", "sources": [{"ref": "n1#7"}, {"ref": "n1#9"}]}, refs)
    assert ans == "Budget was raised [1] and ignored. Also [2]."          # a dropped marker takes one preceding space with it
    assert kept == ["n1#7", "n1#6"]
    src = chat.resolve(root, "n1#7", aliases)
    assert src["speaker"] == "Speaker 2" and src["start"] == 70.0 and src["title"] == "A note" and chat.resolve(root, "n1#99", aliases) is None


def test_prompt_stays_under_ctx():
    import chat, pipeline
    root = Path(tempfile.mkdtemp()); d = make_note(root, "2026-10-09-100000-one", SEGS * 40)  # 1200 turns
    s = {"aliases": {}, "messages": [{"role": "user", "text": "q" * 3000}, {"role": "assistant", "text": "a" * 3000}] * 5, "context": []}
    ranges = chat.union_context([], [{"note": d.name, "first": 0, "last": 1199, "score": 1}], root)
    text, refs = chat.render_context(root, ranges, chat.aliases_for(s, [d.name]))
    msgs = chat.build_messages(s, "final question", text, chat.notes_list(root))
    assert chat.prompt_chars(msgs) <= pipeline.NUM_CTX * pipeline.CHARS_PER_TOKEN * 0.85
    assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": "final question"} and len(msgs) >= 4


def test_claude_argv_and_failure():
    import chat
    argv = chat.claude_argv("hello")
    for f in ["-p", "hello", "--output-format", "json", "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--tools", "--no-session-persistence", "--json-schema"]:
        assert f in argv, f
    assert json.loads(argv[argv.index("--json-schema") + 1]) == chat.ANSWER_SCHEMA and ("--model" in argv) == bool(chat.CLAUDE_MODEL)
    assert chat.flatten([{"role": "system", "content": "S"}, {"role": "user", "content": "ctx"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q"}]) == "User:\nctx\n\nAssistant:\na1\n\nUser:\nq"
    chat.CLAUDE_BIN = "/nonexistent/claude"
    try:
        chat.call_claude([{"role": "user", "content": "q"}]); assert False
    except (RuntimeError, OSError):
        pass


def test_session_roundtrip_and_answer():
    import chat
    root = Path(tempfile.mkdtemp()); d = make_note(root, "2026-10-09-100000-one", SEGS); chat.index_note(d, embed=fake_embed)
    s = chat.new_session(root, None, "local")
    assert chat.list_sessions(root)[0]["id"] == s["id"] and chat.list_sessions(root, note="x") == []
    s = chat.queue_question(root, s["id"], "Budget line pe kya baat hui?", "local")
    assert s["title"] == "Budget line pe kya baat hui?" and chat.pending(s)
    try:
        chat.queue_question(root, s["id"], "again", "local"); assert False
    except ValueError:
        pass
    fake = lambda msgs: {"answer": "Budget pe baat hui [n1#7]. Not recorded [n1#500].", "sources": [{"ref": "n1#7"}, {"ref": "n1#500"}]}
    s = chat.answer(root, s["id"], embed=fake_embed, models={"local": fake})
    m = s["messages"][-1]
    assert not m["pending"] and m["text"] == "Budget pe baat hui [1]. Not recorded." and [x["turn"] for x in m["sources"]] == [7]
    assert m["sources"][0]["title"] == "A note" and s["context"] and s["aliases"] == {"n1": d.name}
    boom = lambda msgs: (_ for _ in ()).throw(RuntimeError("no model"))
    s = chat.queue_question(root, s["id"], "and after that?", "local"); s = chat.answer(root, s["id"], embed=fake_embed, models={"local": boom})
    assert s["messages"][-1]["error"].startswith("RuntimeError") and not chat.pending(s)
    s = chat.queue_question(root, s["id"], "x", "local"); chat.mark_interrupted(root)
    assert chat.load_session(root, s["id"])["messages"][-1]["error"] == "interrupted by a restart"
    chat.clear_session(root, s["id"]); assert chat.load_session(root, s["id"])["messages"] == []
    chat.delete_session(root, s["id"]); assert chat.list_sessions(root) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
