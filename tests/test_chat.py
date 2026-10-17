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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
