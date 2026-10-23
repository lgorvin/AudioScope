"""Minimal Standard MIDI File (SMF type 0) writer — no external dependency.

Notes are given in seconds and converted with the given tempo, so the file plays
back at the original timing and lines up with the grid in a DAW set to that tempo.
Note numbers follow the MIDI standard (60 = middle C; FL Studio labels it C5).
"""
from __future__ import annotations

import struct

from .transcribe import NoteEvent

PPQ = 480  # ticks per quarter note


def _vlq(value: int) -> bytes:
    """MIDI variable-length quantity."""
    value = max(0, int(value))
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(out))


def notes_to_midi(notes: list[NoteEvent], bpm: float | None = None, track_name: str = "Bass",
                  channel: int = 0, program: int | None = 33) -> bytes:
    """Build an SMF type-0 file. ``bpm`` None writes 120 BPM (timing stays correct in seconds).

    ``program`` 33 = General MIDI 'Electric Bass (finger)' (0-based 33); None omits the program change.
    """
    tempo = bpm if bpm and bpm > 0 else 120.0
    ticks_per_s = PPQ * tempo / 60.0
    ch = channel & 0x0F
    events: list[tuple[int, int, bytes]] = []  # (tick, order, data); note-offs sort before note-ons
    for nt in notes:
        on = int(round(nt.start * ticks_per_s))
        off = max(on + 1, int(round(nt.end * ticks_per_s)))
        key = max(0, min(127, int(nt.midi)))
        vel = max(1, min(127, int(nt.velocity)))
        events.append((on, 1, bytes([0x90 | ch, key, vel])))
        events.append((off, 0, bytes([0x80 | ch, key, 0])))
    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    name = track_name.encode("utf-8")[:127]
    track += _vlq(0) + b"\xff\x03" + _vlq(len(name)) + name
    us_per_q = int(round(60_000_000 / tempo))
    track += _vlq(0) + b"\xff\x51\x03" + us_per_q.to_bytes(3, "big")
    track += _vlq(0) + b"\xff\x58\x04" + bytes([4, 2, 24, 8])  # 4/4
    if program is not None:
        track += _vlq(0) + bytes([0xC0 | ch, program & 0x7F])
    last = 0
    for tick, _, data in events:
        track += _vlq(tick - last) + data
        last = tick
    track += _vlq(0) + b"\xff\x2f\x00"

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, PPQ)
    return header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)


def write_midi(path: str, notes: list[NoteEvent], bpm: float | None = None, track_name: str = "Bass") -> None:
    with open(path, "wb") as f:
        f.write(notes_to_midi(notes, bpm, track_name))
