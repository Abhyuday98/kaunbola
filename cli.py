#!/usr/bin/env python3
"""kaunbola CLI (stdlib only). transcribe goes through the server; list/search/show read the notes dir."""
import hashlib, json, os, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

AUDIO_EXT = {".m4a", ".mp3", ".wav", ".ogg", ".opus", ".webm", ".mp4", ".mkv", ".aac", ".flac", ".mov"}

NOTES = Path(os.environ.get("NOTES_DIR", "~/notes/audio")).expanduser()
API = os.environ.get("AUDIO_NOTES_URL", "http://127.0.0.1:8470")


def api(method, path, data=None, headers=None):
    req = urllib.request.Request(API + path, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def submit(src, title="", folder=""):
    """-> (id, duplicate). Skips the upload when the same file was transcribed before."""
    try:
        try:
            return api("GET", "/api/notes/by-hash/" + sha256(src))["id"], True
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        q = urllib.parse.urlencode({"filename": src.name, "title": title, "folder": folder, "created": src.stat().st_mtime})
        r = api("POST", f"/api/upload?{q}", data=src.read_bytes(), headers={"Content-Type": "application/octet-stream"})
        return r["id"], bool(r.get("duplicate"))
    except urllib.error.URLError as e:
        sys.exit(f"server not reachable at {API} ({e}). Start it: systemctl --user start audio-notes")


def opt(args, name):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return ""


def transcribe(args):
    folder = opt(args, "--folder")
    src = Path(args[0]).expanduser()
    nid, dup = submit(src, args[1] if len(args) > 1 else "", folder)
    if dup:
        print(f"already transcribed: {nid}", file=sys.stderr)
    else:
        print(f"queued {nid}", file=sys.stderr)
    while True:
        st = json.loads((NOTES / nid / "status.json").read_text())
        if st["state"] == "done":
            break
        if st["state"] == "error":
            sys.exit(f"error: {st['detail']}")
        print(f"  {st['state']} {st.get('detail', '')}", file=sys.stderr)
        time.sleep(5)
    print((NOTES / nid / "transcript.md").read_text())
    print(f"\n[saved to {NOTES / nid / 'transcript.md'}]", file=sys.stderr)


def import_dir(args):
    """import <dir> [--min-seconds 60]: queue every recording under dir; folder = first sub-directory name."""
    min_s = float(opt(args, "--min-seconds") or 60)
    root = Path(args[0]).expanduser().resolve()
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT and not any(part.startswith(".") for part in p.relative_to(root).parts))
    for p in files:
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
        if not dur or float(dur) < min_s:
            continue
        rel = p.relative_to(root).parts
        folder = rel[0] if len(rel) > 1 else root.name
        nid, dup = submit(p, "", folder)
        print(f"{'already  ' if dup else 'queued   '} {nid}\t{folder}\t{p.name}")


def list_notes(args):
    n = int(args[0]) if args else 30
    dirs = [d for d in NOTES.iterdir() if d.is_dir()]
    for d in sorted(dirs, key=lambda d: load(d / "meta.json").get("created", ""), reverse=True)[:n]:
        t, st = load(d / "transcript.json"), load(d / "status.json")
        meta = load(d / "meta.json")
        names = t.get("speakers", {})
        who = ", ".join(v or ("Speaker " + str(int(k.rsplit("_", 1)[-1]) + 1) if k.startswith("SPEAKER_") else k) for k, v in names.items())
        dur = time.strftime("%H:%M:%S", time.gmtime(t.get("duration", 0)))
        print(f"{d.name}\t{st.get('state', '?')}\t{meta.get('created', '')[:16]}\t{dur}\t{meta.get('folder', '')}\t{meta.get('title') or t.get('title', '')}\t{who}")


def search(args):
    q = " ".join(args)
    r = subprocess.run(["grep", "-rin", "--include=transcript.md", "--", q, str(NOTES)], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        path, _, rest = line.partition(":")
        print(f"{Path(path).parent.name}:{rest[:300]}")


def show(args):
    p = NOTES / args[0] / "transcript.md"
    print(p.read_text() if p.exists() else f"no transcript for {args[0]}: {load(NOTES / args[0] / 'status.json')}")


def speakers(args):
    for name, rec in load(NOTES / "speakers.json").items():
        print(f"{name}\t{len(rec['embeddings'])} samples")


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    cmds = {"transcribe": transcribe, "import": import_dir, "list": list_notes, "search": search, "show": show, "speakers": speakers}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit("usage: cli.py transcribe <file> [title] [--folder F] | import <dir> [--min-seconds 60] | list [n] | search <query> | show <id> | speakers")
    cmds[sys.argv[1]](sys.argv[2:])
