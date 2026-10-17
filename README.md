# turnwise

Turn by turn: who spoke, what they said, in the words they used.

Self-hosted meeting transcription for conversations that switch between Hindi and English mid-sentence. Record on your phone or drop in a file, and your own GPU turns it into a transcript that says who said what, in the words they actually used: Hinglish written in Latin script, an English line under each turn, speakers named once and recognised in every later recording, plus a title, summary and action items. Nothing leaves your machines, unless you choose to have Claude answer a chat question.

I built it after noticing what transcription tools do with conversations that switch between Hindi and English: they either force the whole thing into English, write it in Devanagari, or send the audio to someone else's cloud. What I wanted was narrower than "AI meeting notes": capture who said what, verbatim, in the script people actually type in, and be able to ask questions about it later from Claude Code.

<p>
<img src="docs/app.gif" width="300" alt="The phone app: the home screen with two small monsters taking turns to talk, a transcript with each speaker in their own colour, then a chat question answered from the recordings with the turns it used listed underneath, and one of them playing in a small player with the words on screen">
&nbsp;
<img src="docs/demo.gif" width="560" alt="A short two-person Hinglish meeting transcribed from the command line: speakers, timestamps, the Hinglish line and an English line per turn, summary and action items">
</p>

The app on the left, the command line on the right. The recordings in the clip are made up.

A turn in the transcript looks like this:

```
[00:12:41] **Priya:** Haan toh basically hum log Monday ko dobara milenge, tab tak draft ready ho jayega.
> Yes, so basically we will meet again on Monday; the draft will be ready by then.
```

## What it does

- **Record or upload.** A phone-installable web app (PWA) on your Tailscale network: tap to record, share a file from any app, or upload several large files at once with progress. The note's date comes from the file, not the upload.
- **Noise removal** for phone recordings in cafés and cars, with the attenuation capped so quiet, far-away speakers are not silenced.
- **Speaker detection** with voice memory: name "Speaker 2" once and the same voice is named automatically in future recordings.
- **Hinglish the way people type it**, with an English translation per turn and the raw recognition kept alongside.
- **Title, summary and action items** generated from the content.
- **Folders, a date view, and dedupe by content hash**: re-sending a file you already transcribed just opens it, whatever it is called.
- **A chat that answers from the recordings** and shows the moments it drew on: tap one and it plays, with the words on screen. Across everything or inside one note, with sessions kept. Answered by the local model or, for the owner, by Claude Code on their own subscription.
- **Every voice has a colour.** Each person gets a hue, the same in the transcript, the chat sources and the player, so who said what reads at a glance. Two small monsters on the home screen take turns talking, which is the whole idea.
- **Fast on a phone over a tunnel.** Screens paint before the server answers, going back is instant and keeps your place, and no request hangs: after twenty seconds you get a retry banner.
- **A Claude Code skill** so that "what did we decide about the draft last week?" is answered from the transcripts.
- **One engine** behind the web app, the CLI and the bulk importer, so every path produces the same note.
- **Runs as a service**: starts on boot, frees the GPU when idle, resumes interrupted jobs after a restart.

## How a recording flows through

1. **Convert and clean.** ffmpeg makes a mono WAV. DeepFilterNet3 removes background noise in two-minute chunks in parallel on the CPU, attenuation capped at 30 dB by default. On by default; it can be turned off in Settings, and per job, for clean recordings.
2. **Who spoke when.** pyannote `speaker-diarization-community-1` runs on the GPU and returns exclusive speaker turns plus one voice embedding per speaker.
3. **What was said.** Qwen3-ASR-1.7B transcribes three-minute windows cut at silence, so the model has conversational context, prompted with a romanised Hinglish sentence so English words stay in Latin script. Meta's MMS forced aligner (via torchaudio) then gives every word a timestamp; Devanagari words are romanised with uroman for alignment only. Each word goes to the speaker whose turn contains it, and runs of words become turns. `ASR_MODE=turns` is the simpler alternative: one clip per speaker turn.
4. **Who is this.** Each speaker's embedding is compared (cosine) with the voices stored in `speakers.json`. Above the threshold the name is applied. Naming a speaker in the app stores that voice, keeping the last ten samples per person.
5. **Rewrite and summarise.** A local LLM through Ollama, with structured JSON output, rewrites each turn as Latin-script Hinglish and translates it to English, fifteen turns per request. A second pass writes the title, summary and action items, reducing long transcripts chunk by chunk first. The raw recognition is kept in the note and shown under an "Original" tab.
6. **Store.** `transcript.md` for humans and Claude, `transcript.json` for code, `meta.json` with the SHA-256 of the source file.

Models load lazily, stay resident between jobs, and unload after ten minutes idle. If Ollama runs on the same GPU, the speech models are unloaded before the LLM pass and reloaded on the next job; if it runs on another machine (below), nothing is ever evicted.

### Why these models

| Step | Choice | What I tested |
| --- | --- | --- |
| Speech recognition | Qwen3-ASR-1.7B | faster-whisper large-v3 on the same Hinglish phone clip misdetected the language as Malay and translated instead of transcribing. Kept as `ASR_BACKEND=whisper` for English-only use. |
| Diarization | pyannote community-1 | The best open diarizer, and its output includes the embeddings the voice memory needs. Gated on Hugging Face (free, CC-BY-4.0). |
| Word timing | torchaudio MMS forced aligner | Qwen's own aligner has no Hindi. Window mode with alignment matched per-turn mode on a three-minute clip and keeps context on hour-long meetings. |
| Noise removal | DeepFilterNet3 static binary | No Python dependencies, fast on CPU. Full-strength suppression removed quiet speakers, hence the cap. |
| Hinglish and summaries | Ollama, `qwen3.5:9b` by default | `qwen3.6:35b-a3b` on a second machine ran at the same speed and was slightly more verbatim (it kept a syllable the 9B dropped). Bigger models improve summaries, not the words or the speakers. |

## Setup

Tested on Ubuntu with one RTX A2000 (12 GB), Python 3.12, ffmpeg, and Ollama already installed. A 30-minute recording takes about nine minutes end to end on that card.

```bash
git clone https://github.com/samotherguy/turnwise
cd turnwise
bash scripts/setup.sh        # venv, pinned requirements, DeepFilterNet binary, config template, user service
```

Then:

1. Accept the model terms at [huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), create a read token, and paste it into `~/.config/audio-notes/env`. Without it you still get a transcript, with one speaker.
2. `ollama pull qwen3.5:9b` (or set `LLM_MODEL` to what you have) and `ollama pull bge-m3` for the chat's retrieval.
3. `systemctl --user restart audio-notes`, then `loginctl enable-linger $USER` so it starts on boot.
4. To reach it from your phone, serve it over Tailscale; the browser microphone needs HTTPS:

```bash
tailscale serve --bg --https=8446 http://127.0.0.1:8470
```

The app serves everything itself, including its display font, so the phone talks to your machine and nothing else. Open the address Tailscale gives you on the phone and add it to the home screen. The installed app gets a "Start recording" shortcut on long-press and appears in the share sheet, so a recording made with the phone's own voice recorder can be sent straight to it. For long meetings use the native recorder and share: a browser tab does not survive the screen locking or a phone call.

### A second machine for the LLM

If you have another box on the tailnet, the LLM pass can live there so the transcription GPU is never shared and the model stays loaded:

```bash
# on the second machine
bash scripts/setup-ollama-remote.sh <its-tailscale-ip> qwen3.6:35b-a3b
# on the transcription machine, in ~/.config/audio-notes/env
OLLAMA_URL=http://<its-tailscale-ip>:11434
LLM_MODEL=qwen3.6:35b-a3b
```

The script binds Ollama to that one address only. A 35B mixture-of-experts model with 3B active parameters runs usefully on a 12 GB card with the rest in RAM; a dense model of that size would not.

## Using it

**Web app.** Record, upload, open a note: summary and action items at the top, then the transcript with Hinglish / English / Original tabs, tap a turn to seek the audio, tap a speaker to name them. The "⋯" menu renames, moves to a folder, copies markdown, re-transcribes or deletes. Settings has the voice-match threshold, the denoise toggle, and the list of known voices.

**CLI** (standard library only, talks to the running service):

```bash
.venv/bin/python cli.py transcribe meeting.m4a [title] [--folder F]   # waits and prints the markdown; a known file just prints it
.venv/bin/python cli.py import ~/recordings [--min-seconds 60]       # queue a whole tree; folder = first sub-directory
.venv/bin/python cli.py ask "what did we decide about the draft?" [--note <id>] [--model local|claude]
.venv/bin/python cli.py list | search <words> | show <id> | speakers | reindex
```

**Asking questions.** The Ask button on the home screen (or "Ask about this recording" in a note's menu) opens a chat. Each answer is written only from transcript turns and lists them underneath: who, which recording, when, the Hinglish and the English. Tap one to play that moment in a small player that shows the words turn by turn and links into the full note. Chats are kept as sessions; clear one or start another from its menu. Follow-up questions keep the previous question's material in view, so "and what did she reply?" works.

How it works: every note gets an index of its turns, embedded with `bge-m3` through Ollama together with the note's title (`EMBED_URL`, `EMBED_MODEL`; built when a note finishes and backfilled at startup, or with `cli.py reindex`). A question retrieves the closest chunks, boosts the chunks of any speaker named in the question, adds exact hits for rare words and numbers, and hands the model those turns with references it must cite. References it invents are dropped before you see them; a quoted passage it forgot to cite is matched back to its turn. Every Ollama call shares one context size, so the model keeps its cache between questions. On a 12 GB card the local model answers in 30 to 40 seconds; follow-ups are faster.

The owner can switch a question to Claude. That runs the Claude Code CLI in headless mode on the server with `--setting-sources ""` and no MCP servers, so each question costs a few thousand tokens of overhead rather than the account's whole connector catalogue (`CLAUDE_BIN`, `CLAUDE_MODEL`). Be clear about what that means: the retrieved turns and the question go to Anthropic. The local model never leaves the tailnet, and guests never get the Claude option.

**Claude Code.** Copy `skill/` to `~/.claude/skills/turnwise/` and set the path at the top of it. Claude then transcribes files and answers questions about past meetings through the CLI, never touching the GPU itself.

**Speaker maintenance** (`notes_tool.py`):

- `from-vtt <note_id> <file.vtt> --apply`: if a recording comes with a WebVTT subtitle file that carries speaker names (most video-call platforms produce one for a call), this maps the detected speakers to those names by speaking-time overlap and stores their voices. The best way to seed the voice memory. It does not help when several people share one device, since the platform then credits everything to that one account.
- `rematch --apply [--reset=Name1,Name2]`: relabel unnamed speakers across all notes from the current known voices, optionally reconsidering some names. Manual names are never overwritten.
- `repair --apply`: strip hallucinated echoes of the prompt sentence from old transcripts.

Deleting a known voice also untags that name in every note, so the labels can be redone.

## More than one person

Anyone you add to your tailnet can use the app with their own notes. Tailscale Serve stamps each request with the visitor's login and overwrites any header a client sends, so the server needs no login of its own: requests with your login, or with none (the CLI and the Claude skill run locally), use your notes; every other login gets its own root under `~/notes/audio-users/<login>/` with its own voice memory, settings and theme. A note id only resolves inside the requester's root, so nobody can open anyone else's notes, and the CLI and skill never see the other roots. Set `OWNER_LOGIN` in the env file to your tailnet login.

Everyone shares the one GPU queue. Each person picks their own look in Settings; there is a pink theme besides the classic one.

By default a new tailnet member can reach everything on your machines, not just this app. An access rule in the Tailscale admin console limits them to the one port:

```json
{"action": "accept", "src": ["their@login"], "dst": ["your-machine:8446"]}
```

## Where it is honest about its limits

- **Speaker attribution is the weakest link**, as in every tool of this kind. The diarizer over-splits: one person can come back as two speakers in a long call, and short interjections ("okay", "correct") land in a mixed cluster. Naming a voice in three or four recordings makes matching robust; telling the diarizer how many people were in the room would help more and is not wired up yet.
- **The Hinglish line is a rewrite.** The LLM re-scripts and translates; it does not hear anything. A small model occasionally tidies a filler word. The raw recognition is always kept.
- **Qwen echoes its prompt** on silent stretches, producing the context sentence as if someone said it. These are stripped wherever they appear, and `repair` cleans older notes.
- **No login.** Access control is the tailnet. Do not expose it further without putting something in front.
- **The chat is retrieval, not reading.** The model sees the turns that matched, about two questions' worth, not the whole corpus. A question about something phrased very differently from how it was said can miss; the list of recordings in the prompt lets the model say which recording exists but was not retrieved.
- **Renaming a note** keeps its old title in the search index until the next restart or `cli.py reindex`.
- **One worker, one GPU.** Jobs queue. Fine for one person, not for a team.
- **Tested on one machine and one language pair.** Qwen3-ASR covers 52 languages, but the prompt, context sentence and LLM instructions are written for Hindi and English.

## Future work

Things I would do next, roughly in the order they would help:

- **Speaker-count hint** at upload or record time, passed to the diarizer. The cheapest attribution gain available.
- **Verbatim guard**: when a turn's Hinglish drifts too far in length from the raw recognition, keep the raw words and flag the turn.
- **A/B the Hinglish fine-tune of Qwen3-ASR** ([Srota](https://discuss.huggingface.co/t/open-source-hinglish-asr-fine-tuned-from-qwen3-asr-0-6b-15-85-wer-on-conversational-test/176416), 0.6B) against the 1.7B base on real phone clips. The only open lever on word accuracy.
- **Local LLM fallback** when the remote Ollama is unreachable, so the note still gets its Hinglish lines.
- **Subtitle auto-import**: drop a `.vtt` next to the recording and have names seeded without a command.
- **Second worker** on a second GPU for bulk imports.
- **Streaming answers** in the chat, token by token, instead of waiting for the whole reply.
- **Word-level subtitles** in the player: the aligner already has word times; storing them would make the highlight follow each word.
- **Live transcription** while recording, for the cases where reading along matters more than accuracy.
- **Auth** in front of the app, so it can live beyond a tailnet.

## Licences

Code is MIT. The models have their own terms: pyannote community-1 is CC-BY-4.0 and gated; Qwen3-ASR is Apache-2.0; DeepFilterNet is MIT/Apache-2.0; the MMS forced aligner weights are CC-BY-NC-4.0, which matters if you use this commercially. The Baloo 2 font bundled in `static/fonts/` is under the SIL Open Font License.

## Tests

```bash
.venv/bin/python -I tests/test_chat.py   # index, retrieval, prompt budget, validation, sessions, race on clear/delete
.venv/bin/python pipeline.py             # self-checks for chunking, alignment, context-echo stripping
```

The phone behaviour (chat, player, instant navigation, drafts surviving re-renders) is checked in WebKit with Playwright's iPhone profile; the scripts live outside the repo for now.

## Design notes

