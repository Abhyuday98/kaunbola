"""A/B the Hinglish/English pass on an existing note without re-transcribing.
  OLLAMA_URL=http://host:11434 LLM_MODEL=qwen3.6:35b-a3b .venv/bin/python scripts/ab_llm.py <note_id> [n_turns]
Prints drift stats (hinglish word count vs raw text, Devanagari leaks), speed, and the first turns side by side."""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as pl

nid, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
t = pl.read_json(pl.NOTES / nid / "transcript.json", None)
dev = re.compile(r"[\u0900-\u097F]")
segs = [dict(s) for s in t["segments"] if dev.search(s["text"])][:n]  # only turns with Hindi: English-only turns are copied verbatim by both models
old = [(s.pop("hinglish", ""), s.pop("english", "")) for s in segs]
t0 = time.time()
warn = pl.hinglish_english(segs, lambda d: None)
dt = time.time() - t0
dev = re.compile(r"[ऀ-ॿ]")

def stats(pairs):
    drift = sum(abs(len(h.split()) - len(s["text"].split())) > 0.2 * max(1, len(s["text"].split())) for (h, _), s in zip(pairs, segs))
    leaks = sum(bool(dev.search(h)) for h, _ in pairs)
    return f"drift>20%: {drift}/{len(segs)}  devanagari leaks: {leaks}"

new = [(s.get("hinglish", ""), s.get("english", "")) for s in segs]
print(f"model={pl.LLM_MODEL} at {pl.OLLAMA}  {len(segs)} turns in {dt:.0f}s  warn={warn}")
print("stored (9b):", stats(old))
print("new        :", stats(new))
for s, (oh, _), (nh, ne) in list(zip(segs, old, new))[:20]:
    print(f"\nRAW : {s['text']}\nOLD : {oh}\nNEW : {nh}\nENG : {ne}")
