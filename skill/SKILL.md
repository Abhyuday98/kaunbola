---
name: turnwise
description: Transcribe meeting recordings (Hindi + English + Hinglish, speaker-labelled, noise-removed, with auto title, summary and action items) on the local GPU, and retrieve past transcripts from the notes knowledge base. Use when the user mentions transcribing, a recording, meeting notes, audio/voice/video files (.m4a .mp3 .wav .webm .mp4), "what did X say in the meeting", importing a folder of recordings, or wants to search/recall past meetings.
---

# turnwise

Local meeting transcription (web app on the tailnet, GPU pipeline, this CLI). Set `TURNWISE` to the repo checkout.
Notes live in `~/notes/audio/<id>/` with `transcript.md` (read this), `transcript.json` (structured), `meta.json` (title, folder, sha256), `audio.*`, plus `chunks.json`/`chunks.npy` (the chat's search index; never edit). `~/notes/audio/chats/` holds the user's chat sessions, not notes.

Other people who use the app get their own notes under `~/notes/audio-users/<login>/`. Those are theirs: never list, search, read or mention them unless the user explicitly asks about that person's notes.

Always use the CLI; never load the models yourself (one server process owns the GPU). Every path (web app, CLI, import) runs the same engine
and produces the same output: title generated from content, summary, action items, speaker-labelled Hinglish + English transcript.

```bash
TURNWISE="$HOME/turnwise"            # wherever the repo is checked out
CLI="$TURNWISE/.venv/bin/python $TURNWISE/cli.py"
$CLI transcribe <file> [title] [--folder F]   # uploads, waits, prints transcript.md. If the same file (sha256) was done before it says so and prints it.
$CLI import <dir> [--min-seconds 60]          # queue every recording under dir (folder = first sub-directory); skips already-transcribed files; returns immediately
$CLI list [n]                                 # id, status, duration, folder, title, speakers — newest first
$CLI ask <question> [--note <id>] [--model local|claude]   # answers from the transcripts (all notes, or one) and lists the turns it used as "[k] title HH:MM:SS Speaker: words"
$CLI search <words>                           # grep across all transcript.md, prints "<id>:<line>"
$CLI show <id>                                # prints transcript.md
$CLI speakers                                 # known voices (named by the user in the web app)
$CLI reindex                                  # rebuild the chat's search index (after bulk changes or a rename); runs in the background
```

If the server is unreachable: `systemctl --user start audio-notes` (logs: `journalctl --user -u audio-notes -f`).
Long files take time (~9 min per 30 min of audio); `transcribe` prints progress to stderr while it waits. After `import`, check progress with `list`.

## Transcript format
`transcript.md`: header (title, date, duration, speakers, model), `## Summary`, `## Action items` (`- **Owner:** item`), then `## Transcript` with one line per turn:
`[HH:MM:SS] **Name or Speaker 1:** <Hinglish in Latin script>` followed by `> <English translation>` when they differ.

`transcript.json`: `{id, title, created, duration, folder, summary, action_items: [{owner, item}], speakers: {SPEAKER_00: "Name"|null}, segments: [{start, end, speaker, text (raw ASR), hinglish, english, lang}], warning?}`.

## Answering questions about meetings
1. `ask "<the question>"` (add `--note <id>` to restrict it to one recording). It retrieves the relevant turns, answers only from them, and prints the turns it used, so you can quote them. The local model answers by default; `--model claude` sends the retrieved turns to Anthropic and is the owner's choice, not yours.
2. For reading a whole meeting, `search` for the topic / person, then `show` the matching ids (or read `~/notes/audio/<id>/transcript.md` directly). Summaries and action items are at the top of each file.
3. Quote speakers by name; unnamed ones appear as "Speaker 1/2…". The user names voices in the web app and names propagate to future recordings automatically.

## Speaker maintenance (`notes_tool.py`, run with the repo `.venv`)
- If a recording comes with a WebVTT subtitle file carrying speaker names (video-call platforms produce one for a call), `python notes_tool.py from-vtt <note_id> <file.vtt> --apply` names the note's speakers from it and stores their voices as known speakers. Best way to seed voice memory. (Useless when several people share one device: the platform then credits everything to that one account.)
- `python notes_tool.py rematch --apply [--reset=Name1,Name2] [note_id_to_skip ...]` labels unnamed clusters (and clusters currently carrying a `--reset` name) across all notes from the current known voices; other manual names are kept.
- `python notes_tool.py repair --apply` removes hallucinated context-sentence echoes from old transcripts.
- Deleting a known voice in the UI or via `DELETE /api/speakers/<name>` also untags that name in every note.
