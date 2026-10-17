"""denoise -> diarize -> transcribe -> name speakers -> write note. One job at a time (GPU)."""
import datetime, json, os, re, shutil, subprocess, tempfile, threading, time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
NOTES = Path(os.environ.get("NOTES_DIR", "~/notes/audio")).expanduser()
ASR_BACKEND = os.environ.get("ASR_BACKEND", "qwen")  # qwen | whisper
ASR_MODE = os.environ.get("ASR_MODE", "window")  # window: transcribe 3-min windows for context, align words to speakers | turns: one clip per speaker turn
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
DIAR_MODEL = "pyannote/speaker-diarization-community-1"
EMB_MODEL = DIAR_MODEL  # centroids come from the diarization pipeline
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5:9b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_SHARES_GPU = any(h in OLLAMA for h in ("127.0.0.1", "localhost"))  # remote Ollama: never evict either side
NUM_CTX = 24576  # one context size for every Ollama call: a different num_ctx reloads the model (14 s) and drops its KV cache
CHARS_PER_TOKEN = 2.4  # measured for Hinglish on qwen3.6; prompts longer than NUM_CTX are silently cut in half by Ollama
SR = 16000
DEFAULT_SETTINGS = {"match_threshold": 0.55, "denoise": True, "atten_db": 30, "language": "auto", "theme": "system"}
LANG_NAMES = {"en": "English", "hi": "Hindi"}
# Tested on real Hinglish phone clips: forcing "English" keeps English words in Latin script and Hindi in Devanagari
# (auto-detect writes everything in Devanagari). The romanised context nudges mixed output further.
HINGLISH_CONTEXT = "Haan toh basically hum log yeh recording transcribe kar rahe hain, baaki details baad mein dekhenge."


# ---------- storage ----------
def read_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(p)


def settings(root=NOTES):
    """Per-root (= per person) settings; a note's root is its parent directory."""
    return {**DEFAULT_SETTINGS, **read_json(root / "settings.json", {})}


def speakers_db(root=NOTES):
    return read_json(root / "speakers.json", {})


def set_status(note_dir, state, detail="", started=None):
    write_json(Path(note_dir) / "status.json", {"state": state, "detail": detail, "at": time.time(), "started": started})


def new_note_id(title):
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "note").lower()).strip("-")[:40] or "note"
    return datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S") + "-" + slug


def spk_label(k):
    return f"Speaker {int(k.rsplit('_', 1)[-1]) + 1}" if k.startswith("SPEAKER_") else k


def fmt_ts(s):
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ---------- audio ----------
def ffmpeg(src, dst, rate):
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(rate), str(dst)], capture_output=True, text=True)
    if r.returncode:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip()
        raise RuntimeError("This file has no audio track." if not probe else f"ffmpeg could not decode the file: {r.stderr.strip()[:200]}")


def denoise(wav48, atten_db, status=lambda d: None):
    """deep-filter is single-threaded: run 2-minute chunks in parallel and report progress."""
    from concurrent.futures import ThreadPoolExecutor
    audio, sr = sf.read(wav48, dtype="float32")
    work = Path(wav48).parent / "dfn"
    (work / "out").mkdir(parents=True)
    n = sr * 120
    bounds = [(off, min(off + n, len(audio))) for off in range(0, max(len(audio), 1), n)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < sr * 5:  # deep-filter panics on sub-second files: fold a short tail in
        bounds[-2:] = [(bounds[-2][0], bounds[-1][1])]
    for i, (a, b) in enumerate(bounds):
        sf.write(work / f"p{i:04d}.wav", audio[a:b], sr)

    def run(i):
        subprocess.run([str(ROOT / "bin" / "deep-filter"), "-D", "--atten-lim-db", str(atten_db), "-o", str(work / "out"), str(work / f"p{i:04d}.wav")],
                       check=True, capture_output=True)

    status(f"0/{len(bounds)}")
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as ex:
        for done, _ in enumerate(ex.map(run, range(len(bounds))), 1):
            status(f"{done}/{len(bounds)}")
    def fixed(i, a, b):  # deep-filter drops ~30 ms per file after delay compensation; keep timestamps exact
        y = sf.read(work / "out" / f"p{i:04d}.wav", dtype="float32")[0]
        return np.pad(y, (0, max(0, (b - a) - len(y))))[:b - a]
    clean = np.concatenate([fixed(i, a, b) for i, (a, b) in enumerate(bounds)])
    sf.write(work / "clean.wav", clean, sr)
    return work / "clean.wav"


def prepare_audio(src, work, do_denoise, atten_db, status=lambda d: None):
    """-> path of 16k mono wav."""
    w48 = work / "a48.wav"
    ffmpeg(src, w48, 48000)
    if do_denoise:
        w48 = denoise(w48, atten_db, status)
    w16 = work / "a16.wav"
    ffmpeg(w48, w16, SR)
    return w16


# ---------- models (lazy, resident, idle-unloaded) ----------
class Models:
    IDLE_UNLOAD = 600

    def __init__(self):
        self.lock = threading.Lock()
        self._diar = self._asr = self._align = None
        self.diar_error = None
        self.last_used = time.time()
        threading.Thread(target=self._reaper, daemon=True).start()

    def _reaper(self):
        while True:
            time.sleep(60)
            with self.lock:
                if self._asr and time.time() - self.last_used > self.IDLE_UNLOAD:
                    self.unload()

    def unload(self):
        import torch
        self._diar = self._asr = self._align = None
        torch.cuda.empty_cache()

    def diar(self):
        if self._diar is None:  # retried every job, so accepting the gated model needs no restart
            try:
                if not HF_TOKEN:
                    raise RuntimeError("HF_TOKEN not set")
                import torch
                from pyannote.audio import Pipeline
                self._diar = Pipeline.from_pretrained(DIAR_MODEL, token=HF_TOKEN).to(torch.device("cuda"))
                self.diar_error = None
            except Exception as e:  # gated model not accepted, no token, no network…
                self.diar_error = f"diarization unavailable ({type(e).__name__}: {str(e)[:200]}). Accept {DIAR_MODEL} on huggingface.co and set HF_TOKEN."
        return self._diar


    def asr(self):
        if self._asr is None:
            import torch
            if ASR_BACKEND == "whisper":
                import ctypes, glob
                for lib in ("nvidia/cublas/lib/libcublas.so.12", "nvidia/cudnn/lib/libcudnn.so.9"):  # ctranslate2 wants CUDA 12 libs; torch is cu130
                    for so in glob.glob(str(Path(torch.__file__).parent.parent / lib)):
                        ctypes.CDLL(so)
                from faster_whisper import WhisperModel
                self._asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
            else:
                from qwen_asr import Qwen3ASRModel
                self._asr = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
                                                          max_inference_batch_size=8, max_new_tokens=3072)
        return self._asr

    def aligner(self):
        """Meta MMS forced aligner (torchaudio built-in): 1100+ languages incl. Hindi, on romanised text."""
        if self._align is None:
            import torch, torchaudio
            b = torchaudio.pipelines.MMS_FA
            self._align = (b.get_model(with_star=False).to(torch.device("cuda")).eval(), b.get_tokenizer(), b.get_aligner())
        return self._align


MODELS = Models()


# ---------- steps ----------
def diarize(audio, duration):
    """audio: float32 16k mono -> (turns, {speaker: unit embedding})"""
    pipe = MODELS.diar()
    if pipe is None:
        return [(0.0, duration, "SPEAKER_00")], {}
    import torch
    out = pipe({"waveform": torch.from_numpy(audio)[None, :], "sample_rate": SR})  # in-memory: avoids torchcodec/ffmpeg lib issues
    turns = [(t.start, t.end, spk) for t, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
    embs = {}
    for i, spk in enumerate(out.speaker_diarization.labels()):
        v = np.asarray(out.speaker_embeddings[i], dtype=float)
        if np.isfinite(v).all() and np.linalg.norm(v) > 0:
            embs[spk] = (v / np.linalg.norm(v)).tolist()
    return sorted(turns), embs


def merge_turns(turns, gap=1.0, max_len=30.0, min_len=0.4):
    """Adjacent same-speaker turns -> one clip, capped at max_len seconds."""
    out = []
    for s, e, spk in turns:
        if out and out[-1][2] == spk and s - out[-1][1] < gap and e - out[-1][0] <= max_len:
            out[-1] = (out[-1][0], e, spk)
        else:
            out.append((s, e, spk))
    return [t for t in out if t[1] - t[0] >= min_len]


def split_long(audio, clips, max_len=30.0, search=8.0):
    """Cut clips longer than max_len at the quietest 50 ms frame in the last `search` seconds of each window."""
    out = []
    for s, e, spk in clips:
        while e - s > max_len:
            lo, hi = int((s + max_len - search) * SR), int((s + max_len) * SR)
            frames = audio[lo:hi].reshape(-1, SR // 20)[:, :] if hi - lo >= SR // 20 else None
            cut = (lo + int(np.argmin(np.sqrt((frames ** 2).mean(1)))) * (SR // 20)) / SR if frames is not None else s + max_len
            out.append((s, cut, spk))
            s = cut
        out.append((s, e, spk))
    return out


def transcribe_clips(audio, clips, lang, status=lambda d: None):
    """audio: float32 16k mono. -> [(text, lang)] per clip."""
    n = len(audio)
    pad = int(0.3 * SR)
    arrays = [audio[max(0, int(s * SR) - pad): min(n, int(e * SR) + pad)] for s, e, _ in clips]
    model = MODELS.asr()
    if ASR_BACKEND == "whisper":
        res = []
        for i, a in enumerate(arrays):
            status(f"{i}/{len(arrays)}")
            segs, info = model.transcribe(a, language=lang, beam_size=5, vad_filter=False, condition_on_previous_text=False,
                                          initial_prompt=None if lang == "hi" else HINGLISH_CONTEXT)
            res.append((" ".join(s.text.strip() for s in segs), info.language))
        return res
    res = []
    for i in range(0, len(arrays), 8):
        status(f"{i}/{len(arrays)}")
        results = model.transcribe(audio=[(a, SR) for a in arrays[i:i + 8]], language=LANG_NAMES.get(lang, "English"),
                                   context="" if lang == "hi" else HINGLISH_CONTEXT)
        res += [(strip_context(r.text), r.language) for r in results]
    return res


_UROMAN = None
_ROM_CACHE = {}


def romanize_word(w):
    """-> lowercase [a-z'] form for the aligner ('' if nothing alignable, e.g. punctuation or digits)."""
    global _UROMAN
    if w not in _ROM_CACHE:
        if _UROMAN is None:
            from uroman import Uroman
            _UROMAN = Uroman()
        r = w if w.isascii() else _UROMAN.romanize_string(w)
        _ROM_CACHE[w] = re.sub(r"[^a-z']", "", r.lower())
    return _ROM_CACHE[w]


def align_words(audio, text):
    """-> [(word, start, end)] in seconds relative to `audio` (float32 16k)."""
    import torch
    words = text.split()
    norm = [romanize_word(w) for w in words]
    idx = [i for i, n in enumerate(norm) if n]
    if not idx:
        return []
    model, tokenizer, aligner = MODELS.aligner()
    with torch.inference_mode():
        emission, _ = model(torch.from_numpy(audio)[None, :].to("cuda"))
        spans = aligner(emission[0], tokenizer([norm[i] for i in idx]))
    sec = (len(audio) / SR) / emission.size(1)
    times = {i: (sp[0].start * sec, sp[-1].end * sec) for i, sp in zip(idx, spans)}
    out, last = [], 0.0
    for i, w in enumerate(words):
        if i in times:
            st, en = times[i]
        else:  # unalignable token (digits, symbols): glue to the previous word's end
            st = en = last
        out.append((w, st, en))
        last = en
    return out


def transcribe_windows(audio, lang, status=lambda d: None):
    """Transcribe ~3-minute windows (cut at silence) so the model has conversational context; -> word list."""
    windows = split_long(audio, [(0.0, len(audio) / SR, "")], max_len=180.0, search=20.0)
    model = MODELS.asr()
    words = []
    for i in range(0, len(windows), 4):
        status(f"{i}/{len(windows)} windows")
        batch = windows[i:i + 4]
        arrays = [audio[int(s * SR):int(e * SR)] for s, e, _ in batch]
        results = model.transcribe(audio=[(a, SR) for a in arrays], language=LANG_NAMES.get(lang, "English"),
                                   context="" if lang == "hi" else HINGLISH_CONTEXT)
        for (s, e, _), a, r in zip(batch, arrays, results):
            text = strip_context(r.text)
            if not text:
                continue
            words += [(w, s + ws, s + we, r.language) for w, ws, we in align_words(a, text)]
    return words


def words_to_segments(words, turns, max_gap=2.0, max_len=45.0):
    """Give each word the speaker whose (exclusive) turn contains its midpoint, then group runs into segments."""
    import bisect
    starts = [t[0] for t in turns]
    segs, prev_spk = [], turns[0][2] if turns else "SPEAKER_00"
    for w, ws, we, lang in words:
        mid = (ws + we) / 2
        k = bisect.bisect_right(starts, mid) - 1
        spk = prev_spk
        if k >= 0 and turns[k][0] <= mid <= turns[k][1]:
            spk = turns[k][2]
        else:  # in a gap: nearest turn within 1 s, else keep the previous speaker
            cands = [t for t in (turns[k] if k >= 0 else None, turns[k + 1] if k + 1 < len(turns) else None) if t]
            if cands:
                t = min(cands, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))
                if min(abs(t[0] - mid), abs(t[1] - mid)) <= 1.0:
                    spk = t[2]
        if segs and segs[-1]["speaker"] == spk and ws - segs[-1]["end"] <= max_gap and we - segs[-1]["start"] <= max_len:
            segs[-1]["end"] = max(segs[-1]["end"], we)
            segs[-1]["text"] += " " + w
        else:
            segs.append({"start": ws, "end": we, "speaker": spk, "text": w, "lang": lang})
        prev_spk = spk
    for sg in segs:
        sg["start"], sg["end"] = round(sg["start"], 2), round(sg["end"], 2)
    return segs


CONTEXT_ECHO = [re.compile(r"[^.?!]*\b(recording transcribe kar|transcrib\w+ this recording|baaki details baad mein|rest of the details later|remaining details later)\b[^.?!]*[.?!]?\s*", re.I)]  # Hinglish and English renderings of the context


def strip_context(text):
    """Qwen echoes the context sentence (or a translation of it) on silent/noisy windows; drop any sentence that is it."""
    for rx in CONTEXT_ECHO:
        text = rx.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_speakers(embs, threshold, root=NOTES):
    db = speakers_db(root)
    names = {}
    for spk, v in embs.items():
        best, best_sim = None, threshold
        for name, rec in db.items():
            if rec.get("model") != EMB_MODEL:
                continue
            sim = max(float(np.dot(v, np.asarray(x))) for x in rec["embeddings"])
            if sim > best_sim:
                best, best_sim = name, sim
        names[spk] = best
    return names


def remember_speaker(name, embedding, root=NOTES):
    db = speakers_db(root)
    rec = db.setdefault(name, {"model": EMB_MODEL, "embeddings": []})
    if rec.get("model") != EMB_MODEL:
        rec.update(model=EMB_MODEL, embeddings=[])
    rec["embeddings"] = (rec["embeddings"] + [embedding])[-10:]  # ponytail: keep last 10 samples per person
    write_json(root / "speakers.json", db)


# ---------- hinglish + english via local LLM ----------
LLM_PROMPT = """You convert meeting-transcript segments. Input is a JSON list of {i, text}; text is Hindi (Devanagari), English, or a mix.
For each item return {i, hinglish, english}:
- hinglish: the SAME words in Latin script, the way Indians type Hinglish. English words stay as they are; Hindi words are romanised naturally (मैं -> main, क्या -> kya, नहीं -> nahi). The hinglish field must contain no Devanagari characters at all (हाँ -> haan, ठीक है -> theek hai). If the text is already fully English, copy it unchanged. Never translate English into Hindi.
- english: a faithful English translation (copy unchanged if already English).
Keep every sentence and every item. Do not summarise, merge, or add anything."""


def llm_messages(messages, schema, num_predict=None):
    import urllib.request
    opts = {"temperature": 0.1, "num_ctx": NUM_CTX, **({"num_predict": num_predict} if num_predict else {})}
    body = {"model": LLM_MODEL, "stream": False, "think": False, "keep_alive": "5m" if LLM_SHARES_GPU else -1, "format": schema,
            "options": opts, "messages": messages}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def llm_json(system, user, schema):
    return llm_messages([{"role": "system", "content": system}, {"role": "user", "content": user}], schema)


def ollama_unload():
    """Free the LLM's VRAM before loading the GPU models (it reloads in ~8 s for the translate step)."""
    import urllib.request
    if not LLM_SHARES_GPU:
        return
    try:
        req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps({"model": LLM_MODEL, "keep_alive": 0}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass


def llm_batch(items):
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
        "i": {"type": "integer"}, "hinglish": {"type": "string"}, "english": {"type": "string"}}, "required": ["i", "hinglish", "english"]}}}, "required": ["items"]}
    out = llm_json(LLM_PROMPT, json.dumps(items, ensure_ascii=False), schema)["items"]
    return {o["i"]: o for o in out}


SUMMARY_SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"}, "summary": {"type": "string"},
    "action_items": {"type": "array", "items": {"type": "object", "properties": {"owner": {"type": "string"}, "item": {"type": "string"}}, "required": ["owner", "item"]}}},
    "required": ["title", "summary", "action_items"]}
SUMMARY_PROMPT = """You write meeting notes from a transcript (English translation of a Hindi/English meeting; speakers may be unnamed).
Return JSON: title (max 8 words, specific to what was discussed, no quotes), summary (3-8 sentences or short bullet lines in markdown: what was discussed, decisions, open questions), action_items (concrete follow-ups; owner = speaker name or role if clear, else ""). Only include what is actually in the transcript."""
CHUNK_SCHEMA = {"type": "object", "properties": {"key_points": {"type": "array", "items": {"type": "string"}},
                "action_items": {"type": "array", "items": {"type": "string"}}}, "required": ["key_points", "action_items"]}


def summarize(segments, speakers, status=lambda d: None):
    """-> {title, summary, action_items}. Long transcripts are reduced chunk by chunk first."""
    lines = [f"[{fmt_ts(s['start'])}] {speakers.get(s['speaker']) or spk_label(s['speaker'])}: {s.get('english') or s['text']}" for s in segments]
    text = "\n".join(lines)
    CH = int(NUM_CTX * CHARS_PER_TOKEN * 0.6)  # ~35k chars: room left for the prompt and the answer inside NUM_CTX
    if len(text) > CH:
        parts = [text[i:i + CH] for i in range(0, len(text), CH)]
        notes = []
        for i, part in enumerate(parts):
            status(f"part {i + 1}/{len(parts)}")
            r = llm_json("Extract the key points and action items from this part of a meeting transcript. Be concrete and faithful.", part, CHUNK_SCHEMA)
            notes.append("Key points:\n- " + "\n- ".join(r["key_points"]) + "\nAction items:\n- " + "\n- ".join(r["action_items"]))
        text = "Notes from consecutive parts of one meeting:\n\n" + "\n\n".join(notes)
    status("writing")
    return llm_json(SUMMARY_PROMPT, text, SUMMARY_SCHEMA)


def hinglish_english(segments, status):
    """Fills seg['hinglish'] / seg['english'] in place. Returns a warning string or ''."""
    todo = [(i, s) for i, s in enumerate(segments) if s["text"]]
    B = 15
    try:
        for k in range(0, len(todo), B):
            status(f"{k}/{len(todo)}")
            batch = todo[k:k + B]
            got = llm_batch([{"i": i, "text": s["text"]} for i, s in batch])
            for i, s in batch:
                o = got.get(i, {})
                s["hinglish"] = o.get("hinglish") or s["text"]
                s["english"] = o.get("english") or s["text"]
    except Exception as e:
        for _, s in todo:
            s.setdefault("hinglish", s["text"]); s.setdefault("english", s["text"])
        return f"Hinglish/English pass failed ({type(e).__name__}: {str(e)[:120]}); is Ollama running with {LLM_MODEL}?"
    return ""


# ---------- output ----------
def write_md(note_dir, note):
    names = note["speakers"]
    lines = [f"# {note['title']}", "",
             f"- Date: {note['created'][:16].replace('T', ' ')}",
             f"- Duration: {fmt_ts(note['duration'])}",
             f"- Speakers: {', '.join(names.get(k) or spk_label(k) for k in names)}",
             f"- Transcribed by: {note['backend']}" + ("  (" + note["warning"] + ")" if note.get("warning") else ""),
             ""]
    if note.get("summary"):
        lines += ["## Summary", "", note["summary"].strip(), ""]
    if note.get("action_items"):
        lines += ["## Action items", ""] + [f"- {'**' + a['owner'] + ':** ' if a.get('owner') else ''}{a['item']}" for a in note["action_items"]] + [""]
    lines += ["## Transcript", ""]
    for seg in note["segments"]:
        who = names.get(seg["speaker"]) or spk_label(seg["speaker"])
        lines.append(f"[{fmt_ts(seg['start'])}] **{who}:** {seg.get('hinglish') or seg['text']}")
        if seg.get("english") and seg["english"] != seg.get("hinglish"):
            lines.append(f"> {seg['english']}")
    (Path(note_dir) / "transcript.md").write_text("\n".join(lines) + "\n")


def run(note_dir):
    note_dir = Path(note_dir)
    meta = read_json(note_dir / "meta.json", {})
    root = note_dir.parent
    cfg = {**settings(root), **meta.get("options", {})}
    src = next(p for p in note_dir.iterdir() if p.stem == "audio")
    started = time.time()
    st = lambda state, detail="": set_status(note_dir, state, detail, started)
    try:
        with MODELS.lock, tempfile.TemporaryDirectory() as tmp:
            MODELS.last_used = time.time()
            work = Path(tmp)
            ollama_unload()
            st("denoise" if cfg["denoise"] else "convert")
            wav16 = prepare_audio(src, work, cfg["denoise"], cfg["atten_db"], lambda d: st("denoise", d))
            audio, _ = sf.read(wav16, dtype="float32")
            duration = len(audio) / SR

            st("diarize")
            turns, embs = diarize(audio, duration)
            clips = split_long(audio, merge_turns(turns))

            st("transcribe", f"0/{len(clips)}")
            lang = None if cfg["language"] == "auto" else cfg["language"]
            texts = transcribe_clips(audio, clips, lang, lambda d: st("transcribe", d))

            spk_ids = sorted({c[2] for c in clips})
            names = match_speakers(embs, cfg["match_threshold"], root) if embs else {}
            MODELS.last_used = time.time()
            segments = [{"start": round(s, 2), "end": round(e, 2), "speaker": spk, "text": t, "lang": l}
                        for (s, e, spk), (t, l) in zip(clips, texts) if t]
            st("translate")
            if LLM_SHARES_GPU:
                MODELS.unload()  # ponytail: free VRAM for Ollama; ASR reloads (~20 s) on the next job
            warning = hinglish_english(segments, lambda d: st("translate", d))
            st("summarize")
            spk_names = {s: names.get(s) for s in spk_ids}
            try:
                summ = summarize(segments, spk_names, lambda d: st("summarize", d)) if segments else {}
            except Exception as e:
                summ, warning = {}, (warning + " " if warning else "") + f"Summary failed ({type(e).__name__}: {str(e)[:120]})."
            if not segments:
                peak = float(np.abs(audio).max()) if len(audio) else 0.0
                warning = "No speech detected" + (" — the audio track is silent (screen recording without a microphone?)." if peak < 0.01 else ".")
            if summ.get("title") and meta.get("title_auto", True):
                meta["title"] = summ["title"].strip()
                write_json(note_dir / "meta.json", meta)

        note = {"id": note_dir.name, "title": meta.get("title") or note_dir.name, "created": meta.get("created") or datetime.datetime.now().isoformat(timespec="seconds"),
                "duration": round(duration, 1), "language": cfg["language"],
                "backend": ("qwen3-asr-1.7b" + ("+mms-align" if ASR_MODE == "window" else "")) if ASR_BACKEND == "qwen" else "whisper-large-v3",
                "speakers": spk_names, "folder": meta.get("folder", ""),
                "summary": summ.get("summary", ""), "action_items": summ.get("action_items", []),
                "embeddings": embs, "segments": segments}
        warning = " ".join(w for w in (MODELS.diar_error, warning) if w)
        if warning:
            note["warning"] = warning
        write_json(note_dir / "transcript.json", note)
        write_md(note_dir, note)
        st("done", note.get("warning", ""))
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        if "out of memory" in msg.lower():
            msg += " — GPU is full; if Ollama has a model loaded run `ollama stop <model>` and retry."
        st("error", msg)
        raise


if __name__ == "__main__":
    import sys
    a = np.ones(SR * 70, dtype=np.float32); a[SR * 28:SR * 28 + SR // 20] = 0  # quiet frame at 28 s
    assert [round(c[1], 2) for c in split_long(a, [(0, 70, "A")])] == [28.0, 50.0, 70]
    turns = [(0, 5, "A"), (5, 8, "B"), (8, 12, "A")]
    words = [("hi", 0.1, 0.4, "en"), ("there", 0.5, 0.9, "en"), ("yes", 5.2, 5.5, "en"), ("ok", 8.3, 8.6, "en"), ("2026", 8.6, 8.6, "en")]
    segs = words_to_segments(words, turns)
    assert [(x["speaker"], x["text"]) for x in segs] == [("A", "hi there"), ("B", "yes"), ("A", "ok 2026")], segs
    assert romanize_word("हाँ") == "haam" and romanize_word("client,") == "client" and romanize_word("2026") == ""
    assert strip_context(HINGLISH_CONTEXT + " So, hamko kya dalna hai") == "So, hamko kya dalna hai"
    assert strip_context("Kal ka plan. toh toh basically hum log yeh recording transcribe kar rahe hain, baaki details baad mein dekhenge. These are") == "Kal ka plan. These are"
    assert strip_context("Basically, we are transcribing this recording. We will see the rest of the details later.") == ""
    assert strip_context("Okay. Details baad mein bhejna.") == "Okay. Details baad mein bhejna."
    assert merge_turns([(0, 1, "A"), (1.2, 2, "A"), (2.5, 3, "B"), (3.1, 3.2, "B")]) == [(0, 2, "A"), (2.5, 3.2, "B")]
    assert NUM_CTX == 24576 and "num_ctx" not in llm_json.__code__.co_varnames
    assert summarize.__code__.co_consts.count(12288) == 0 and summarize.__code__.co_consts.count(30000) == 0
    if len(sys.argv) < 2:
        print('self-checks ok'); sys.exit()
    # smoke check: python pipeline.py some.m4a  -> transcribes into NOTES and prints the markdown
    src = Path(sys.argv[1])
    nd = NOTES / new_note_id(src.stem)
    nd.mkdir(parents=True)
    shutil.copy(src, nd / ("audio" + src.suffix.lower()))
    write_json(nd / "meta.json", {"title": src.stem, "created": datetime.datetime.now().isoformat(timespec="seconds"), "options": {}})
    t0 = time.time()
    run(nd)
    print((nd / "transcript.md").read_text())
    print(f"--- {time.time() - t0:.0f}s, status:", read_json(nd / "status.json", {}))
