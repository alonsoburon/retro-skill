"""Tests for skills/retro/scripts/opencode-transcript.py.

Builds a tiny opencode database under `tempfile` and asserts the JSONL the adapter
renders. The load-bearing case is the LAST one: the detector is run over the
rendered transcript, because a `tool_use` block without an `id` makes it raise
`KeyError: 'id'` on the first tool call — which is exactly how the mapping was
discovered, by running it rather than by reading it.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


def _load(name: str, filename: str):
    """Import a hyphenated script by path, the way `test_detect_mechanical.py` does."""
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "skills" / "retro" / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = _load("opencode_transcript", "opencode-transcript.py")
detector = _load("detect_mechanical", "detect-mechanical.py")


def _database(path: str) -> None:
    """A real opencode database, only as wide as the adapter reads."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, data TEXT)"
    )
    conn.execute(
        "CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT,"
        " time_created INTEGER, data TEXT)"
    )
    rows = [
        ("m1", "s1", 1, {"role": "user"}),
        ("m2", "s1", 2, {"role": "assistant"}),
    ]
    for message_id, session_id, created, data in rows:
        conn.execute(
            "INSERT INTO message VALUES (?,?,?,?)",
            (message_id, session_id, created, json.dumps(data)),
        )
    parts = [
        ("p1", "m1", "s1", {"type": "text", "text": "please fix the CLI"}),
        ("p2", "m2", "s1", {"type": "text", "text": "looking"}),
        (
            "p3",
            "m2",
            "s1",
            {
                "type": "tool",
                "tool": "bash",
                "id": "call-1",
                "state": {
                    "status": "completed",
                    "input": {"command": "ls"},
                    "output": "boom",
                },
            },
        ),
    ]
    for index, (part_id, message_id, session_id, data) in enumerate(parts, start=1):
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?,?)",
            (part_id, message_id, session_id, index, json.dumps(data)),
        )
    conn.commit()
    conn.close()


class OpencodeTranscriptTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = str(Path(tmp.name) / "opencode.db")
        _database(self.db)

    def test_a_tool_part_becomes_a_tool_use_and_a_matching_tool_result(self) -> None:
        conn = adapter._connect(self.db)
        lines = [json.loads(line) for line in adapter.render(conn, "s1")]

        kinds = [
            block["type"] for line in lines for block in line["message"]["content"]
        ]
        self.assertIn("tool_use", kinds)
        self.assertIn("tool_result", kinds)

        uses = [
            b
            for line in lines
            for b in line["message"]["content"]
            if b["type"] == "tool_use"
        ]
        results = [
            b
            for line in lines
            for b in line["message"]["content"]
            if b["type"] == "tool_result"
        ]
        self.assertEqual(uses[0]["name"], "bash")
        self.assertEqual(uses[0]["id"], "call-1")
        # The detector pairs them BY ID, so a missing one is the whole defect.
        self.assertEqual(results[0]["tool_use_id"], uses[0]["id"])

    def test_the_session_is_found_by_content_and_an_unknown_token_is_refused(
        self,
    ) -> None:
        conn = adapter._connect(self.db)
        self.assertEqual(adapter.find_session(conn, "fix the CLI"), "s1")
        with self.assertRaises(SystemExit):
            adapter.find_session(conn, "a token that is nowhere")

    def test_a_path_carrying_a_query_cannot_override_the_read_only_mode(self) -> None:
        """`mode=ro` is what the READ-ONLY promise rests on; an unencoded path drops it."""
        hostile = str(Path(self.db).parent / "opencode.db?mode=rwc&")
        Path(hostile).write_bytes(Path(self.db).read_bytes())

        conn = adapter._connect(hostile)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE injected (x)")

    def test_the_detector_accepts_the_rendered_transcript(self) -> None:
        """The shape is only right if the CONSUMER takes it: no `KeyError: 'id'`."""
        conn = adapter._connect(self.db)
        out = Path(self.db).parent / "session.jsonl"
        out.write_text("\n".join(adapter.render(conn, "s1")) + "\n", encoding="utf-8")

        events = detector.load_jsonl(out)
        self.assertTrue(events)
        self.assertEqual(len(detector.extract_tool_uses(events)), 1)


if __name__ == "__main__":
    unittest.main()
