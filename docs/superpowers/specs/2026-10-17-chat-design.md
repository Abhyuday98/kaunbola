# Chat over recordings, with playable sources

Design note, 2026-10-17. Status: draft for approval.

## Intent

A chat inside the app that answers questions from the transcripts and shows, under each answer, the exact moments it drew on. Tapping a moment plays that audio with the words shown as it plays. The chat exists on the home screen, where it looks across every recording the person owns, and inside a note, where it only looks at that meeting.

The owner's words on what matters: "proper answer", "maintain history", "let me clear or start new session", past sessions kept in a list, "answers need to be as accurate as possible", and a choice per question between the local model and Claude, where Claude runs on the owner's subscription on this machine.

## Goals

- Ask a question in natural language, English or Hinglish, and get a full answer written only from the transcripts.
- Every answer carries a list of sources: note, time, speaker, Hinglish line, English line. Each is playable in place, with turn-by-turn subtitles, and links to the full note at that spot.
- Multi-turn sessions that remember earlier questions and answers. Sessions persist per person, are listed newest first, can be cleared or deleted, and a new one can be started any time.
- Scope: all recordings (home) or one recording (inside a note).
- Model choice: the local Ollama model, or Claude through the Claude Code CLI in headless mode. Claude is offered only to the owner.
- Works on the phone over the tunnel with the same resilience as the rest of the app: an answer that takes a minute must survive the screen locking.

## Non-goals

- Streaming the answer token by token. The answer appears when it is done; a visible "Thinking…" state covers the wait.
- Word-level (karaoke) subtitles. Subtitles change per turn; word timings are not stored today. Accepted by the owner.
- Voice input, sharing chats, editing answers, a separate semantic-search page. The chat is the search.
- Changing how transcription works.

## Anti-goals (what would make this a failure even if it technically works)

- Answers that state things no turn supports, or that soften "not in the recordings" into a guess. The prompt must say so when the material is not there.
- Sources that do not resolve to a real turn, or that resolve to a turn that does not say what the answer claims.
- Any guest's notes reaching Claude, or any person seeing or querying another person's notes or chats.
- A chat that stalls the transcription queue, or a 60-second request that dies when the phone locks.

## Constraints and what was measured

- Corpus today: 25 notes, 6,878 turns, about 225,000 words of Hinglish plus English. It grows by several hours of audio a week. It does not fit in a prompt; retrieval is required.
- Local model: `qwen3.6:35b-a3b` on server 2 over Ollama (`OLLAMA_URL`, `LLM_MODEL` in the env file), one slot, 11.2 of 12.3 GB VRAM in use at `num_ctx` 24576. Measured: prefill runs at about 320 tokens/s regardless of context size (16,800 tokens in 54 s), generation is fast. A follow-up whose prompt starts with the same text as the previous one prefilled in 2.2 s instead of 54 s: the KV cache is reused for the shared prefix. Two things break that reuse, both measured: a request with a different `num_ctx` reloads the model (14 s) and empties the cache; and a prompt longer than `num_ctx` is silently cut to half its length from the front, which shifts the prefix. The pipeline's own calls use 8192 and 12288 today, so every transcription job would evict the chat's cache unless all calls share one `num_ctx`.
- Hinglish tokenises at about 2.4 characters per token with this model, not the 4 the pipeline assumes (`CH = 30000  # chars ≈ 8k tokens` in `summarize` is really about 12,500 tokens, which at `num_ctx` 12288 means long summaries are already being cut). Budgets below are in characters at 2.4 per token.
- Embedding model: `bge-m3` is already pulled on server 1's own Ollama (`http://127.0.0.1:11434`), which is running with the GPU idle. It is multilingual (Hindi and English in one space), 1024 dimensions, and fits beside the speech models on server 1. Server 2's GPU is full with the 35B, so embeddings do not go there.
- Claude: `claude` 2.1.302 is installed and authenticated for the owner. Headless call measured at about 4,000 tokens of fixed overhead when run from an empty directory with `--setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --tools "" --no-session-persistence`. Without those flags the overhead was 142,000 tokens (the account's MCP connectors), so the flags are mandatory.
- Identity: Tailscale Serve stamps the login; `root_for(request)` already resolves each person's notes root; the owner is `OWNER_LOGIN` or a request with no login (localhost, CLI, skill).
- The app is one HTML file with hash routes and a render loop that preserves scroll on in-place re-renders; dialogs come from the in-app `ui` helpers; the theme is CSS tokens, so the pink theme applies to new screens for free.

## Decided design

### Index

Every note gets a chunk index next to its transcript, in the note directory:

- `chunks.json`: a list of chunks, each `{first, last, start, end, speakers, text}` where `first`/`last` are turn indexes into `segments`.
- `chunks.npy`: a float32 matrix, one row per chunk, L2-normalised bge-m3 embeddings.

Chunking is by turns: consecutive turns are grouped until roughly 600 characters of Hinglish, with one turn of overlap between neighbours. The text embedded is the Hinglish lines joined by newlines, using the raw `text` when `hinglish` is missing. Speaker names are not embedded, so renaming or forgetting a speaker never requires re-indexing; the chunk's `speakers` field carries the labels and the model sees names in the rendered context. English lines are not embedded (bge-m3 handles the Hinglish directly and the English would double the index for little gain).

The index is built as a final step of `pipeline.run()` after summarisation, so new notes are searchable the moment they finish. For existing notes, the server builds missing indexes in a background thread at startup (this is network calls to the local Ollama, not GPU work in the transcription worker, so it does not block the queue). `cli.py reindex` rebuilds everything.

At query time all chunk matrices for the requester's root (or the one note) are loaded and searched by brute-force cosine similarity. At today's size that is under 1,000 rows; at fifty times today's size it is still milliseconds. No vector database.

### Retrieval

For a question:

1. Query text is the question. If the session already has messages and the question is short or starts with a pronoun or connective (builder's heuristic), the previous user question is prepended to the query text so follow-ups like "and what did she say after that" retrieve sensibly.
2. Embed with bge-m3. Take the top 24 chunks by cosine.
3. Also run the existing grep search for exact tokens in the question that look like names, numbers or dates, and add any chunk containing a hit. This catches proper nouns the embedding blurs.
4. Expand each selected chunk by two turns on either side, merge overlapping ranges within a note, and order by note then time.
5. Cap the context at about 7,000 tokens (17,000 characters), dropping the lowest-scoring ranges first.
6. In-note scope: if the whole note fits in the cap, send the whole note and skip steps 2 to 5.

Within a session the context block is the union of everything retrieved so far in that session, in the order it was first added, so the prompt prefix stays stable across follow-ups and the local model's KV cache is reused (measured: 2 s instead of 50 s). New ranges for a new question are appended. When the union exceeds the cap, the oldest ranges are dropped from the front; that one follow-up pays a full prefill and the cache is warm again from the next. No "session full" state is needed. The ranges in the context are stored on the session.

### Prompt and answer format

The context is rendered as lines `[note_key#turn] HH:MM:SS Speaker: hinglish`, grouped under a header per note giving its title, date and summary. `note_key` is a short alias (n1, n2, …) mapped on the session, so ids stay short. The home-scope prompt also lists every note title and date so the model knows what exists and can say when a topic was never recorded.

The system prompt instructs: answer in full, in the language of the question; use only the turns given; quote the speaker's words when attributing a statement; if the turns do not contain the answer, say so plainly; cite every claim.

Both models return structured JSON (Ollama `format` schema, Claude `--json-schema`):

```
{"answer": "markdown text with [1], [2] markers",
 "sources": [{"ref": "n1#42"}, {"ref": "n1#43"}, {"ref": "n3#7"}]}
```

The server validates every `ref` against the context it sent; refs that were not in the context are dropped and the corresponding markers removed. Each surviving source is resolved to `{note, title, created, turn, start, end, speaker, hinglish, english}` and stored with the message. The answer text is rendered with the app's existing markdown helper.

### Models

- `local`: Ollama chat on `OLLAMA_URL` with `LLM_MODEL`, temperature 0.1, `think` off, `num_predict` 2048, `keep_alive` as the pipeline already decides. Reuses `pipeline.llm_json` or a thin variant of it that takes the full message list.
- One `NUM_CTX` for every Ollama call, pipeline and chat alike, set to 24576 (fits: 11.2 GB VRAM measured, and the summariser stops being cut). `llm_json` loses its per-call `num_ctx` argument. The prompt builder must keep every prompt under `NUM_CTX` tokens at 2.4 characters per token, since Ollama cuts longer prompts in half without an error: context 17,000 characters, history capped at 8,000 characters (oldest exchanges dropped from the history, never from the context), answer 2048 tokens, system and summaries the rest.
- `claude`: subprocess `claude -p <prompt> --output-format json --json-schema <schema> --setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --tools "" --no-session-persistence`, cwd an empty directory under the config dir, 180 s timeout. The whole conversation (history plus context plus question) is sent in the prompt each time; no `--resume`. Optional `CLAUDE_MODEL` env passed as `--model`. Offered and accepted only for the owner; a guest asking for it gets 403.

The model is chosen per question in the UI; the session remembers the last choice as its default.

On a setup where Ollama shares the GPU with transcription (`LLM_SHARES_GPU`), a local-model chat would evict the speech models mid-job. Builder's call how to handle it; the minimum is to wait until the worker is idle before calling Ollama, with a "waiting for the transcription to finish" state visible in the chat.

### Sessions

Stored per root at `<root>/chats/<id>.json`. Today `note_dirs(root)` treats every subdirectory of a root as a note and `startup()` deletes any subdirectory without a `meta.json`, which would delete the chats on the first restart. Note ids always start with a date (`YYYY-MM-DD-HHMMSS-`), so `note_dirs` is changed to return only directories whose name matches that pattern; `chats` is then invisible to listing, hashing, folders, speaker untagging, the CLI and the startup cleanup.

On startup any assistant message still `pending` from before the restart is marked with `error: "interrupted by a restart"`. Asking in a note that is not `done` returns 409.

```
{"id", "created", "updated", "title", "note": null | note_id, "model": "local"|"claude",
 "aliases": {"n1": note_id, ...}, "context": [refs...],
 "messages": [{"role": "user", "text", "ts"},
              {"role": "assistant", "text", "sources": [...], "model", "ts", "pending": true|false, "error": null|str}]}
```

`title` is the first question, trimmed to 60 characters. Listing is by `updated`, newest first; the in-note list shows only that note's sessions.

Asking is asynchronous: `POST /api/chats/{id}/ask` appends the user message and a pending assistant message, starts the work on a small thread pool separate from the transcription worker, and returns at once. The client polls `GET /api/chats/{id}` every two seconds while a message is pending, the same way the note view polls a running job. This survives the phone locking, the tunnel dropping, and the app being reopened. Only one pending question per session; a second ask while one is pending is rejected.

### Endpoints

All resolve under `root_for(request)`.

- `GET /api/chats?note=<id>`: list sessions (id, title, note, model, updated, message count).
- `POST /api/chats {note?, model?}`: create, returns the session.
- `GET /api/chats/{id}`: full session.
- `POST /api/chats/{id}/ask {text, model?}`: queue a question. 409 if one is pending, 403 if `model` is `claude` and the requester is not the owner.
- `POST /api/chats/{id}/clear`: drop messages and context, keep the session.
- `DELETE /api/chats/{id}`.
- `GET /api/notes/{id}` already returns segments; the player uses it.
- `GET /api/me` gains nothing; `owner` is already there and gates the Claude option in the UI.

### Screens

- **Home bar** gets a third button, Ask, beside Upload and Record, leading to `#/chats`.
- **Note ⋯ menu** gets "Ask about this recording", leading to `#/chats/<note_id>`, whose list is scoped to that note and whose new sessions are scoped to it.
- **Chats list** (`#/chats` or `#/chats/<note_id>`): nav back, title "Chats" or the note's title, a prominent "New chat" row, then sessions as rows with title, date and scope. Swipe or ⋯ to delete is not required; delete lives in the chat's own menu.
- **Chat** (`#/chat/<id>`): nav back to the list, session title, ⋯ menu with New chat, Clear, Delete. Under the nav a segmented control Local / Claude (Claude only rendered for the owner). Messages: user bubbles on the right in the tint colour, answers on the left on a card. Under each answer, the sources as rows: a round ▶ button, a meta line "Speaker · note title · date · mm:ss", the Hinglish line, the English line in secondary colour. Inline `[n]` markers in the answer are tappable and scroll to the matching source row. A pending answer shows a "Thinking…" row with the elapsed seconds. The input is pinned at the bottom with a send button; Enter sends; the input is disabled while a question is pending.
- **Player**: tapping ▶ on a source opens a dark card pinned above the input. It holds a hidden `<audio>` for that note, seeks to the source's start and plays. It shows: speaker and current time, the note title as a link to `#/note/<id>?t=<seconds>`, the Hinglish line of the turn containing the current time, the English line under it, ◀◀ and ▶▶ to jump a turn back or forward, play/pause, a scrub bar, and ✕. The source row currently playing is highlighted. Playing a different source re-uses the player (swaps the audio src if the note changes). Playback continues while the user scrolls the chat. Leaving the chat stops it.
- **Note view** honours `?t=<seconds>` on open by seeking and scrolling to that turn; the player's title link uses this.

Everything uses the existing tokens and `ui` helpers, so the pink theme, dark mode and the 44 px tap targets come along.

### CLI and skill

`cli.py ask "<question>" [--note <id>] [--model local|claude]` creates a session, asks, polls, prints the answer and the sources as `[n] title mm:ss Speaker: hinglish`. `cli.py reindex` builds missing or all indexes. The skill documents `ask` as the way to answer "what did X say about Y" and keeps the rule that other people's roots are never touched.

### Settings and env

- `EMBED_URL` (default `http://127.0.0.1:11434`) and `EMBED_MODEL` (default `bge-m3`).
- `CLAUDE_MODEL` optional. `CLAUDE_BIN` optional, default `claude` on PATH of the service user (the systemd unit's PATH must include `~/.npm-global/bin`).
- Setup script and README gain the embedding model pull and the Claude note.

## What is left to the builder

- Exact chunk size, overlap, top-k and the follow-up query heuristic; tune on the owner's notes with a handful of real questions before calling it done.
- Whether the grep leg uses the existing `/api/search` code path or a direct scan of `transcript.json`.
- How the thread pool is sized and how a crash mid-answer is surfaced (the pending message must end up with `error` set, never stuck).
- The behaviour on a shared-GPU setup (see Models).
- Markdown-to-HTML for the answer: the existing `md` helper covers bold, paragraphs and bullets; extend only if the models produce more.
- Playwright WebKit check of the chat and player on the iPhone profile, as was done for the last phone fixes.

## Builder check, 2026-10-17

A reviewer read the draft and asked one question: is the KV-cache reuse real on this deployment, and what is dropped when a session's context exceeds the cap. Both are now measured and decided above (one shared `NUM_CTX`, union context with oldest-first dropping). The reviewer's hazard about `<root>/chats/` being swept by `note_dirs` and the startup cleanup is handled in Sessions. Its other points (no re-index on speaker rename, 409 on an untranscribed note, pending messages marked as errors on restart) are adopted. No questions remain for the owner.
