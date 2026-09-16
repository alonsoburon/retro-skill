#!/usr/bin/env python3
"""Render an opencode session as the JSONL shape `detect-mechanical.py` reads.

    python3 opencode-transcript.py --match "<a token from the session>" \\
        [--db ~/.local/share/opencode/opencode.db] [--session <id>] > session.jsonl

WHY THIS EXISTS. `detect-mechanical.py` parses a Claude Code transcript, and an
agent running under opencode keeps its session somewhere else entirely: a SQLite
database, `~/.local/share/opencode/opencode.db`, with the messages in `message.data`
and their blocks in `part.data` as JSON. Without this, layer A simply cannot run on
those sessions — the operator is left doing the LLM pass by hand, which is the cost
this project exists to remove.

THE SESSION IS FOUND BY CONTENT. `--match` takes any token from the session under
review and greps the parts for it, exactly as `references/workflow.md` requires of
the Claude path: several sessions share one project, so the newest row is regularly
somebody else's. `--session` skips the search when the id is already known.

THE MAPPING, and the two places it had to be discovered by measuring:

  · a `text` part becomes a `text` block;
  · a `tool` part holds BOTH the call and its result, so it becomes a `tool_use`
    block on the assistant turn AND a `tool_result` block on a user turn
    immediately after — which is where Claude puts it;
  · every `tool_use` NEEDS an `id` and every `tool_result` a matching `tool_use_id`.
    Without them `detect-mechanical.py` raises `KeyError: 'id'` on its first tool
    block, so the id falls back through `callID` to the part's own row id.

READ-ONLY. The database is opened with `mode=ro`, so pointing this at a live
database cannot corrupt a session that is still being written.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

DEFAULT_DB = "~/.local/share/opencode/opencode.db"
#: A block of a very large tool output is worth keeping for the friction signals
#: (an error is at the top, a stack trace at the bottom) and not worth carrying
#: whole: layer A only reads snippets, and a session's outputs run to megabytes.
RESULT_CHARS = 6000


def _connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(f"opencode-transcript: no database at {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def find_session(conn: sqlite3.Connection, token: str) -> str:
    """The session whose parts carry `token`, or a refusal naming the ambiguity."""
    rows = conn.execute(
        "SELECT DISTINCT session_id FROM part WHERE data LIKE ?", (f"%{token}%",)
    ).fetchall()
    if not rows:
        raise SystemExit(f"opencode-transcript: no session carries {token!r}")
    if len(rows) > 1:
        names = ", ".join(r[0] for r in rows)
        raise SystemExit(f"opencode-transcript: {token!r} matches {len(rows)} sessions: {names}")
    return rows[0][0]


def render(conn: sqlite3.Connection, session_id: str) -> list[str]:
    messages = conn.execute(
        "SELECT id, data, time_created FROM message WHERE session_id=? ORDER BY time_created",
        (session_id,),
    ).fetchall()
    parts: dict[str, list[tuple[str, dict]]] = {}
    for row_id, message_id, data, _ts in conn.execute(
        "SELECT id, message_id, data, time_created FROM part WHERE session_id=? ORDER BY time_created",
        (session_id,),
    ):
        parts.setdefault(message_id, []).append((row_id, json.loads(data)))

    lines: list[str] = []
    for message_id, data, timestamp in messages:
        role = json.loads(data).get("role")
        if role not in ("user", "assistant"):
            continue
        blocks: list[dict] = []
        results: list[dict] = []
        for row_id, part in parts.get(message_id, []):
            kind = part.get("type")
            if kind == "text":
                text = part.get("text") or ""
                if text.strip():
                    blocks.append({"type": "text", "text": text})
            elif kind == "tool":
                state = part.get("state") or {}
                call = part.get("call") or {}
                name = part.get("tool") or state.get("tool") or call.get("tool") or "tool"
                payload = state.get("input") or call.get("input") or {}
                tool_id = part.get("callID") or part.get("id") or row_id
                if role == "assistant":
                    blocks.append(
                        {"type": "tool_use", "id": tool_id, "name": name, "input": payload}
                    )
                output = state.get("output")
                if output is None:
                    output = state.get("error") or ""
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": str(output)[:RESULT_CHARS],
                        "is_error": state.get("status") in ("error", "failed"),
                    }
                )
        if blocks:
            lines.append(
                json.dumps(
                    {"type": role, "message": {"role": role, "content": blocks}, "timestamp": timestamp},
                    ensure_ascii=False,
                )
            )
        if results:
            lines.append(
                json.dumps(
                    {"type": "user", "message": {"role": "user", "content": results}, "timestamp": timestamp},
                    ensure_ascii=False,
                )
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opencode-transcript", description=__doc__)
    parser.add_argument("--match", help="a token from the session under review")
    parser.add_argument("--session", help="the opencode session id, when it is known")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output", help="write here instead of stdout")
    args = parser.parse_args(argv)

    if not args.match and not args.session:
        parser.error("pass --match <token> or --session <id>")

    conn = _connect(os.path.expanduser(args.db))
    session_id = args.session or find_session(conn, args.match or "")
    lines = render(conn, session_id)
    text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
