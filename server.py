"""FastAPI app: upload / record / list / read / rename speakers. One GPU worker thread.

Who is asking: Tailscale Serve stamps every proxied request with the visitor's login (and overwrites any
client-supplied header). No header means a local request (CLI, Claude skill) and counts as the owner.
The owner's notes live in NOTES_DIR; every other login gets its own root under USERS_DIR with its own
voice memory and settings, and a note id only ever resolves inside the requester's root.
"""
import datetime, hashlib, mimetypes, os, queue, re, shutil, subprocess, threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

import pipeline as pl

app = FastAPI()
mimetypes.add_type("application/manifest+json", ".webmanifest")
JOBS: "queue.Queue[Path]" = queue.Queue()
EXT = {"audio/webm": ".webm", "audio/mp4": ".m4a", "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/x-m4a": ".m4a", "audio/aac": ".aac"}
OWNER_LOGIN = os.environ.get("OWNER_LOGIN", "").strip().lower()
USERS = Path(os.environ.get("USERS_DIR", str(pl.NOTES.parent / "audio-users"))).expanduser()
UNSAFE = re.compile(r"[^A-Za-z0-9@._-]")
THEMES = ("system", "pink")


def worker():
    while True:
        nd = JOBS.get()
        try:
            pl.run(nd)
        except Exception:
            pass  # status.json already carries the error


def login_of(request):
    return request.headers.get("tailscale-user-login", "").strip().lower()


def root_for(request):
    login = login_of(request)
    if not login or login == OWNER_LOGIN:
        return pl.NOTES
    r = USERS / UNSAFE.sub("_", login)
    r.mkdir(parents=True, exist_ok=True)
    return r


def all_roots():
    return [pl.NOTES] + ([d for d in USERS.iterdir() if d.is_dir()] if USERS.is_dir() else [])


def note_dir(root, nid):
    d = root / nid
    if not d.is_dir() or "/" in nid or nid.startswith("."):
        raise HTTPException(404)
    return d


def hash_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


NOTE_ID = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{6}-")  # note ids are dated; anything else in a root (chats/, speakers.json) is not a note


def note_dirs(root):
    return [d for d in sorted(root.iterdir(), reverse=True) if d.is_dir() and NOTE_ID.match(d.name)]


def find_by_hash(root, h):
    return next((d.name for d in note_dirs(root) if pl.read_json(d / "meta.json", {}).get("sha256") == h), None)


def summary(d):
    meta = pl.read_json(d / "meta.json", {})
    t = pl.read_json(d / "transcript.json", {})
    st = pl.read_json(d / "status.json", {"state": "queued"})
    if not meta:  # upload still streaming in (meta.json is written when the file has fully arrived)
        size = sum(p.stat().st_size for p in d.iterdir())
        st = {"state": "uploading", "detail": f"{size / 1e6:.0f} MB received", "started": d.stat().st_mtime}
        meta = {"title": " ".join(d.name.split("-")[4:]).title() or d.name, "created": datetime.datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")}
    names = t.get("speakers", {})
    return {"id": d.name, "title": meta.get("title") or t.get("title") or d.name, "created": meta.get("created", ""),
            "folder": meta.get("folder", ""), "duration": t.get("duration"), "status": st,
            "speakers": [names.get(k) or pl.spk_label(k) for k in names], "warning": t.get("warning"), "summary": t.get("summary", "")}


def enqueue(d, title=None, options=None, **extra):
    meta = pl.read_json(d / "meta.json", {})
    meta.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    if title:
        meta["title"] = title
    meta.setdefault("title", d.name)
    meta["options"] = options or {}
    meta.update(extra)
    pl.write_json(d / "meta.json", meta)
    pl.set_status(d, "queued")
    JOBS.put(d)


@app.on_event("startup")
def startup():
    pl.NOTES.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    for root in all_roots():
        for d in note_dirs(root):  # resume jobs interrupted by a restart; drop orphaned partial uploads; backfill hashes
            if not (d / "meta.json").exists():
                shutil.rmtree(d)
                continue
            st = pl.read_json(d / "status.json", {})
            if st.get("state") not in (None, "done", "error", "recording") or (st.get("state") == "error" and "SIGTERM" in st.get("detail", "")):
                JOBS.put(d)
            meta = pl.read_json(d / "meta.json", {})
            src = next((p for p in d.iterdir() if p.stem == "audio"), None)
            if src and not meta.get("sha256"):
                meta["sha256"] = hash_file(src)
                pl.write_json(d / "meta.json", meta)


async def write_body(request, path, mode="wb"):
    with open(path, mode) as f:
        async for chunk in request.stream():
            f.write(chunk)


@app.get("/api/me")
def me(request: Request):
    login = login_of(request)
    return {"login": login, "name": request.headers.get("tailscale-user-name", ""), "owner": root_for(request) == pl.NOTES}


@app.post("/api/upload")
async def upload(request: Request, filename: str, title: str = "", folder: str = "", created: float = None, language: str = None, denoise: bool = None):
    """created: the recording's own timestamp (file mtime, epoch seconds); upload time is only the fallback."""
    root = root_for(request)
    d = root / pl.new_note_id(title or Path(filename).stem)
    d.mkdir(parents=True)
    ext = Path(filename).suffix.lower() or ".bin"
    await write_body(request, d / f"audio{ext}")
    h = hash_file(d / f"audio{ext}")
    dup = find_by_hash(root, h)
    if dup:
        shutil.rmtree(d)
        return {"id": dup, "duplicate": True}
    opts = {k: v for k, v in {"language": language, "denoise": denoise}.items() if v is not None}
    when = datetime.datetime.fromtimestamp(created).isoformat(timespec="seconds") if created and created > 1e9 else datetime.datetime.now().isoformat(timespec="seconds")
    enqueue(d, title or Path(filename).stem, opts, title_auto=not title, folder=folder.strip(), sha256=h, source=filename, created=when)
    return {"id": d.name}


@app.get("/api/notes/by-hash/{h}")
def by_hash(request: Request, h: str):
    nid = find_by_hash(root_for(request), h)
    if not nid:
        raise HTTPException(404)
    return {"id": nid}


@app.get("/api/folders")
def folders(request: Request):
    out = {}
    for d in note_dirs(root_for(request)):
        f = pl.read_json(d / "meta.json", {}).get("folder", "")
        out[f] = out.get(f, 0) + 1
    return out


@app.post("/api/record/start")
def record_start(request: Request, title: str = "", folder: str = "", mime: str = "audio/webm"):
    auto = not title
    title = title or "Recording " + datetime.datetime.now().strftime("%d %b %H:%M")
    d = root_for(request) / pl.new_note_id(title)
    d.mkdir(parents=True)
    pl.write_json(d / "meta.json", {"title": title, "title_auto": auto, "folder": folder.strip(), "created": datetime.datetime.now().isoformat(timespec="seconds"), "options": {}, "ext": EXT.get(mime.split(";")[0], ".webm")})
    pl.set_status(d, "recording")
    return {"id": d.name}


@app.post("/api/record/{nid}/chunk")
async def record_chunk(request: Request, nid: str):
    d = note_dir(root_for(request), nid)
    await write_body(request, d / ("audio" + pl.read_json(d / "meta.json", {}).get("ext", ".webm")), "ab")
    return {"ok": True}


@app.post("/api/record/{nid}/finish")
def record_finish(request: Request, nid: str, language: str = None, denoise: bool = None):
    d = note_dir(root_for(request), nid)
    meta = pl.read_json(d / "meta.json", {})
    opts = {k: v for k, v in {"language": language, "denoise": denoise}.items() if v is not None}
    enqueue(d, None, opts, sha256=hash_file(d / ("audio" + meta.get("ext", ".webm"))))
    return {"id": nid}


@app.get("/api/notes")
def list_notes(request: Request):
    return sorted((summary(d) for d in note_dirs(root_for(request))), key=lambda n: n["created"], reverse=True)


@app.get("/api/notes/{nid}")
def get_note(request: Request, nid: str):
    d = note_dir(root_for(request), nid)
    t = pl.read_json(d / "transcript.json", None)
    t = {k: v for k, v in t.items() if k != "embeddings"} if t else {}
    return {**summary(d), **t}


@app.get("/api/notes/{nid}/audio")
def get_audio(request: Request, nid: str):
    d = note_dir(root_for(request), nid)
    p = next((p for p in d.iterdir() if p.stem == "audio"), None)
    if not p:
        raise HTTPException(404)
    return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream")


@app.get("/api/notes/{nid}/md")
def get_md(request: Request, nid: str):
    return PlainTextResponse((note_dir(root_for(request), nid) / "transcript.md").read_text())


@app.patch("/api/notes/{nid}")
def patch_note(request: Request, nid: str, body: dict):
    root = root_for(request)
    d = note_dir(root, nid)
    t = pl.read_json(d / "transcript.json", None)
    if t is None:
        raise HTTPException(409, "not transcribed yet")
    meta = pl.read_json(d / "meta.json", {})
    if body.get("title", "").strip():
        t["title"] = meta["title"] = body["title"].strip()
        meta["title_auto"] = False
    if "folder" in body:
        t["folder"] = meta["folder"] = (body["folder"] or "").strip()
    pl.write_json(d / "meta.json", meta)
    for spk, name in (body.get("speakers") or {}).items():
        name = (name or "").strip() or None
        t["speakers"][spk] = name
        if name and spk in t.get("embeddings", {}):
            pl.remember_speaker(name, t["embeddings"][spk], root)
    pl.write_json(d / "transcript.json", t)
    pl.write_md(d, t)
    return get_note(request, nid)


@app.delete("/api/notes/{nid}")
def delete_note(request: Request, nid: str):
    shutil.rmtree(note_dir(root_for(request), nid))
    return {"ok": True}


@app.post("/api/notes/{nid}/retranscribe")
def retranscribe(request: Request, nid: str, body: dict = None):
    enqueue(note_dir(root_for(request), nid), None, body or {})
    return {"id": nid}


@app.get("/api/search")
def search(request: Request, q: str):
    if not q.strip():
        return []
    r = subprocess.run(["grep", "-ril", "--include=transcript.md", "--", q, str(root_for(request))], capture_output=True, text=True)
    hits = []
    for line in r.stdout.splitlines():
        d = Path(line).parent
        snippet = subprocess.run(["grep", "-i", "-m", "3", "--", q, line], capture_output=True, text=True).stdout.strip()
        hits.append({**summary(d), "snippet": snippet[:400]})
    return sorted(hits, key=lambda h: h["created"], reverse=True)


@app.get("/api/settings")
def get_settings(request: Request):
    return {**pl.settings(root_for(request)), **me(request),
            "diarization": pl.MODELS.diar_error or ("ok" if pl.HF_TOKEN else "HF_TOKEN not set"), "asr_backend": pl.ASR_BACKEND, "llm": pl.LLM_MODEL}


@app.put("/api/settings")
def put_settings(request: Request, body: dict):
    root = root_for(request)
    if body.get("theme") not in (None, *THEMES):
        raise HTTPException(400, "unknown theme")
    pl.write_json(root / "settings.json", {**pl.settings(root), **{k: body[k] for k in pl.DEFAULT_SETTINGS if k in body}})
    return get_settings(request)


@app.get("/api/speakers")
def speakers(request: Request):
    return {name: len(rec["embeddings"]) for name, rec in pl.speakers_db(root_for(request)).items()}


@app.delete("/api/speakers/{name}")
def delete_speaker(request: Request, name: str):
    root = root_for(request)
    db = pl.speakers_db(root)
    db.pop(name, None)
    pl.write_json(root / "speakers.json", db)
    for d in note_dirs(root):  # untag them everywhere, so the labels can be redone
        t = pl.read_json(d / "transcript.json", None)
        if t and name in (t.get("speakers") or {}).values():
            t["speakers"] = {k: (None if v == name else v) for k, v in t["speakers"].items()}
            pl.write_json(d / "transcript.json", t)
            pl.write_md(d, t)
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(pl.ROOT / "static"), html=True))
