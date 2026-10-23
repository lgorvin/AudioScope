"""Command-line analysis: ``python -m audioscope.cli file_or_folder ... [--json] [--type TYPE]``."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

from .engine import AudioType, analyze_file
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
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    results, failed = [], 0
    for path in _expand(args.paths):
        try:
            r = analyze_file(path, AudioType(args.type) if args.type else None)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{os.path.basename(path)}: ERROR {e}", file=sys.stderr)
            continue
        if args.json:
            results.append(r.to_dict())
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
    if args.json:
        json.dump(results, sys.stdout, indent=1)
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
