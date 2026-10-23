"""Command-line analysis: ``python -m audioscope.cli file_or_folder ... [--json] [--type TYPE]``."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

from .engine import AudioType, analyze_file, quantize, transcribe_file, write_midi
from .engine.types import format_bpm


def _expand(paths: list[str]) -> list[str]:
    from .engine.io import is_supported

    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _d, files in os.walk(p):
                out += [os.path.join(root, f) for f in sorted(files) if is_supported(f)]
        else:
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="audioscope", description="Detect BPM, key and root note of audio files.")
    ap.add_argument("paths", nargs="+", help="audio files or folders")
    ap.add_argument("--type", choices=[t.value for t in AudioType], help="skip classification and use this type")
    ap.add_argument("--json", action="store_true", help="print full JSON results")
    ap.add_argument("--bass-midi", metavar="DIR", help="also transcribe the bass line and write DIR/<name>_bass.mid")
    ap.add_argument("--quantize", action="store_true", help="snap MIDI notes to 16ths at the detected tempo")
    args = ap.parse_args(argv)
    if args.bass_midi:
        os.makedirs(args.bass_midi, exist_ok=True)
    warnings.filterwarnings("ignore")

    results, failed = [], 0
    for path in _expand(args.paths):
        try:
            r = analyze_file(path, AudioType(args.type) if args.type else None)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{os.path.basename(path)}: ERROR {e}", file=sys.stderr)
            continue
        bass_line = ""
        if args.bass_midi:
            tr = transcribe_file(path, r.classification.audio_type)
            if tr.notes:
                bpm = r.tempo.bpm
                notes = quantize(tr.notes, bpm) if args.quantize and bpm else tr.notes
                stem = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(args.bass_midi, f"{stem}_bass.mid")
                write_midi(out, notes, bpm, track_name=f"{stem} bass")
                bass_line = f"  bass: {len(notes)} note{'s' if len(notes) != 1 else ''} {tr.range_text} [{tr.reliability.label}] -> {out}"
            else:
                bass_line = "  bass: no notes (" + (tr.evidence[-1] if tr.evidence else "nothing pitched") + ")"
        if args.json:
            d = r.to_dict()
            if args.bass_midi:
                d["bass"] = tr.to_dict()
            results.append(d)
            continue
        t, k, p = r.tempo, r.key, r.pitch
        bpm = f"{format_bpm(t.bpm)} BPM" if t.bpm else "BPM N/A"
        key = f"{k.key.name} ({k.key.camelot})" if k.key else "key N/A"
        root = p.chord or p.root_name or "N/A"
        print(f"{os.path.basename(path)}\n  {r.classification.audio_type.label} [{r.classification.reliability.label}]"
              f"  |  {bpm} [{t.reliability.label}]  |  {key} [{k.reliability.label}]  |  root {root} [{p.reliability.label}]")
        alts = [f"{format_bpm(c.bpm)} ({c.relation})" for c in t.candidates if c.relation != "primary"][:3]
        if alts:
            print("  tempo alternatives: " + ", ".join(alts))
        if k.key is not None and k.candidates[1:]:  # no "alternatives" to a key that is N/A
            print("  key alternatives:   " + ", ".join(f"{c.name} ({c.camelot})" for c in k.candidates[1:4]))
        if bass_line:
            print(bass_line)
    if args.json:
        json.dump(results, sys.stdout, indent=1)
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
