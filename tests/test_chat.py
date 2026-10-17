import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def test_note_dirs_skips_chats():
    import server
    root = Path(tempfile.mkdtemp())
    (root / "2026-10-17-120000-a-note").mkdir(); (root / "chats").mkdir(); (root / "speakers.json").write_text("{}")
    assert [d.name for d in server.note_dirs(root)] == ["2026-10-17-120000-a-note"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
