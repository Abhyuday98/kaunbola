"""Note maintenance.

  python notes_tool.py from-vtt <note_id> <subtitles.vtt> [--apply]
      Map diarized clusters to the speaker names in a WebVTT file (<v Name> cues) by speaking-time overlap.
      --apply names them via the API (which also stores the voices as known speakers).
  python notes_tool.py rematch [--apply] [--reset=Name1,Name2] [note_id_to_skip ...]
      Label unnamed clusters (and clusters currently labelled with a --reset name) from the known voices.
  python notes_tool.py repair [--apply]
      Remove hallucinated context-sentence echoes from existing transcripts.
"""
import json, re, sys, urllib.request
import numpy as np
import pipeline as pl

API = "http://127.0.0.1:8470"
CUE = re.compile(r"(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)\s*\n<v ([^>]+)>", re.M)


def vtt_cues(path):
    for m in CUE.finditer(open(path, encoding="utf-8").read()):
        h, mi, s, ms, h2, mi2, s2, ms2 = (int(x) for x in m.groups()[:8])
        yield h * 3600 + mi * 60 + s + ms / 1000, h2 * 3600 + mi2 * 60 + s2 + ms2 / 1000, m.group(9).strip()


def map_from_vtt(segments, cues, min_share=0.8, min_seconds=5):
    overlap = {}  # cluster -> name -> seconds
    for seg in segments:
        for cs, ce, name in cues:
            o = min(seg["end"], ce) - max(seg["start"], cs)
            if o > 0:
                overlap.setdefault(seg["speaker"], {}).setdefault(name, 0.0)
                overlap[seg["speaker"]][name] += o
    names = {}
    for spk, by_name in overlap.items():
        total = sum(by_name.values())
        best = max(by_name, key=by_name.get)
        share = by_name[best] / total
        names[spk] = best.split()[0] if total >= min_seconds and share >= min_share else None  # ponytail: first name only, matches how notes are labelled
        print(f"{spk}: {total:6.1f}s  {best} {share:.0%}  ->  {names[spk]}")
    return names


def from_vtt(nid, vtt, apply):
    t = pl.read_json(pl.NOTES / nid / "transcript.json", None)
    cues = list(vtt_cues(vtt))
    names = map_from_vtt(t["segments"], cues)
    names = {spk: names.get(spk) for spk in t["speakers"]}
    if apply:
        req = urllib.request.Request(f"{API}/api/notes/{nid}", data=json.dumps({"speakers": names}).encode(),
                                     headers={"Content-Type": "application/json"}, method="PATCH")
        urllib.request.urlopen(req).read()
        print("applied; known voices:", {k: len(v["embeddings"]) for k, v in pl.speakers_db().items()})


def rematch(apply, skip=(), reset=()):
    """Fill unnamed clusters and re-decide clusters currently labelled with a `--reset` name; other manual names are kept."""
    thr = pl.settings()["match_threshold"]
    db = pl.speakers_db()
    for d in sorted(p for p in pl.NOTES.iterdir() if p.is_dir()):
        t = pl.read_json(d / "transcript.json", None)
        if d.name in skip or not t or not t.get("embeddings"):
            continue
        new = pl.match_speakers(t["embeddings"], thr)
        changed = {}
        for spk, emb in t["embeddings"].items():
            cur = t["speakers"].get(spk)
            if cur is not None and cur not in reset:
                continue
            sims = {n: max(float(np.dot(emb, x)) for x in r["embeddings"]) for n, r in db.items()}
            if cur != new[spk]:
                changed[spk] = f"{cur}->{new[spk]} " + " ".join(f"{n}={v:.2f}" for n, v in sims.items())
        print(d.name, changed or "unchanged")
        if apply and changed:
            t["speakers"].update({k: new[k] for k in changed})
            pl.write_json(d / "transcript.json", t)
            pl.write_md(d, t)


def repair(apply):
    for d in sorted(p for p in pl.NOTES.iterdir() if p.is_dir()):
        t = pl.read_json(d / "transcript.json", None)
        if not t:
            continue
        kept, dropped, trimmed = [], 0, 0
        for sg in t["segments"]:
            before = sg["text"]
            for k in ("text", "hinglish", "english"):
                if sg.get(k):
                    sg[k] = pl.strip_context(sg[k])
            if not sg["text"]:
                dropped += 1
                continue
            trimmed += sg["text"] != before
            kept.append(sg)
        print(d.name, f"dropped={dropped} trimmed={trimmed}")
        if apply and (dropped or trimmed):
            t["segments"] = kept
            pl.write_json(d / "transcript.json", t)
            pl.write_md(d, t)


if __name__ == "__main__":
    a = sys.argv[1:]
    apply = "--apply" in a
    if a[0] == "from-vtt":
        from_vtt(a[1], a[2], apply)
    elif a[0] == "repair":
        repair(apply)
    elif a[0] == "rematch":
        reset = next((x.split("=", 1)[1].split(",") for x in a if x.startswith("--reset=")), ())
        rematch(apply, skip=[x for x in a[1:] if not x.startswith("--")], reset=reset)
    else:
        print(__doc__)
