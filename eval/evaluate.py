"""Accuracy evaluation, reported separately for full songs, loops and one-shots.

Two data sources:

* ``--synthetic N``  a seeded random corpus rendered by tests/synth.py (random
  tempos, keys, styles, detuning, MP3 round-trips, added noise). Ground truth
  is exact, but the audio is synthetic: these numbers are an upper bound
  for real recordings.
* ``--dataset DIR --labels labels.csv``  your own labelled audio. CSV columns
  (all optional except file): ``file,category,type,bpm,key,root``, where
  category is song|loop|oneshot, key like "A minor"/"F#m"/"8A" or "none",
  root like "F#2" or "C", and bpm "none" if no tempo applies.

Metrics
-------
BPM: exact (|err| <= 0.5 BPM), Acc1 (+-4 %), Acc2 (+-4 % allowing x2, x1/2, x3, x1/3),
     correct-N/A rate on material without tempo.
Key: exact, MIREX weighted score (1 / 0.5 fifth / 0.3 relative / 0.2 parallel).
Root: pitch-class and exact-note accuracy, median cents error.
Calibration: accuracy per reported reliability level (High should beat Medium beats Low).

Usage::

    python eval/evaluate.py --synthetic 40 --workers 4
    python eval/evaluate.py --dataset D:/refs --labels D:/refs/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import sys
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audioscope.engine import music  # noqa: E402

STYLES_CLEAR = ["house", "trap", "boombap", "techno", "reggaeton"]
STYLES_AMBIGUOUS = ["dnb", "hiphop16"]  # half/double-time ambiguous by construction


# ------------------------------------------------------------------ corpus
def make_synthetic(n: int, out_dir: str, seed: int = 7) -> list[dict]:
    from tests import synth

    rng = random.Random(seed)
    items: list[dict] = []

    def save(y, name, fmt="wav", noise_db=None):
        y = np.asarray(y, dtype=np.float32)
        if noise_db is not None:
            y = y + np.random.default_rng(len(items)).standard_normal(len(y)).astype(np.float32) * 10 ** (noise_db / 20)
        path = os.path.join(out_dir, f"{len(items):03d}_{name}.{fmt}")
        sf.write(path, np.clip(y, -1, 1), synth.SR, format=fmt.upper())
        return path

    def variant_fmt():
        r = rng.random()
        return ("mp3", None) if r < 0.25 else ("wav", -40.0) if r < 0.4 else ("wav", None)

    for i in range(n):
        fmt, noise = variant_fmt()
        kind = i % 4
        tonic, mode = rng.randrange(12), rng.choice(["major", "minor"])
        if kind == 0:
            style = rng.choice(STYLES_CLEAR + STYLES_AMBIGUOUS)
            bpm = rng.randrange(150, 178) if style == "dnb" else rng.randrange(72, 160)
            bars = rng.choice([1, 2, 4, 4, 8])
            y = synth.drum_loop(bpm, bars, style, seed=i, humanize_ms=rng.choice([0, 0, 3]))
            items.append(dict(path=save(y, f"drums_{style}_{bpm}", fmt, noise), category="loop", type="drum_loop",
                              bpm=bpm, key=None, root=None, ambiguous=style in STYLES_AMBIGUOUS, variant=style))
        elif kind == 1:
            bpm, var = rng.randrange(70, 170), rng.choice([0, 1])
            y = synth.chord_loop(bpm, tonic, mode, bars=rng.choice([2, 4, 8]), rhythm=rng.choice(["sustain", "stabs"]),
                                 seed=i, detune_cents=rng.uniform(-35, 35), variant=var)
            items.append(dict(path=save(y, f"chords_{bpm}", fmt, noise), category="loop", type="melodic_loop", bpm=bpm,
                              key=(tonic, mode), root=None, variant=f"progression{var}"))
        elif kind == 2:
            bpm, var = rng.randrange(80, 160), rng.choice([0, 1])
            y = synth.bass_loop(bpm, tonic, mode, bars=4, seed=i, variant=var)
            items.append(dict(path=save(y, f"bass_{bpm}", fmt, noise), category="loop", type="bass_loop", bpm=bpm,
                              key=(tonic, mode), root=None, variant=f"progression{var}"))
        else:
            sub = rng.choice(["note", "note", "808", "chord", "kick", "hat", "vocal"])
            if sub == "note":
                midi, cents = rng.randrange(28, 86), rng.uniform(-40, 40)
                y = synth.one_shot_note(midi, detune_cents=cents, harmonics=rng.choice([1, 4, 8]))
                items.append(dict(path=save(y, f"note_{midi}", fmt, noise), category="oneshot", type="one_shot",
                                  bpm="none", key="none", root=midi, cents=cents, variant="note"))
            elif sub == "808":
                midi = rng.randrange(26, 44)
                y = synth.eight08(midi, dur=rng.uniform(1.0, 3.0), glide_semitones=rng.choice([0, 7, 12, 24]))
                items.append(dict(path=save(y, f"808_{midi}", fmt, noise), category="oneshot", type="one_shot",
                                  bpm="none", key="none", root=midi, variant="808"))
            elif sub == "chord":
                q = rng.choice(["maj", "min", "maj7", "min7"])
                root = rng.randrange(45, 62)
                y = synth.chord_stab(root, q)
                items.append(dict(path=save(y, f"stab_{q}", fmt, noise), category="oneshot", type="one_shot",
                                  bpm="none", key=None, root=root % 12, root_pc_only=True, variant="chord"))
            elif sub == "kick":
                y = synth.kick(f1=rng.uniform(40, 60))[:30000]
                items.append(dict(path=save(y, "kick", fmt, noise), category="oneshot", type="one_shot",
                                  bpm="none", key="none", root=None, variant="kick"))
            elif sub == "hat":
                y = synth.hat(decay=rng.uniform(0.03, 0.2), rng=np.random.default_rng(i))
                items.append(dict(path=save(y, "hat", fmt, noise), category="oneshot", type="one_shot",
                                  bpm="none", key="none", root="none", variant="hat"))
            else:
                y = synth.vocal_phrase(tonic, mode, seconds=rng.uniform(4, 9), seed=i)
                items.append(dict(path=save(y, "vocal", fmt, noise), category="loop", type="vocal", bpm=None,
                                  key=(tonic, mode), root=None, variant="vocal"))
    # full songs (fewer: ~2 min each)
    for j in range(max(2, n // 8)):
        bpm = rng.randrange(80, 150)
        tonic, mode = rng.randrange(12), rng.choice(["major", "minor"])
        var = rng.choice([0, 1])
        style = rng.choice(STYLES_CLEAR)
        y = synth.full_song(bpm, tonic, mode, style=style, seed=100 + j, variant=var)
        fmt = "mp3" if j % 3 == 0 else "wav"
        items.append(dict(path=save(y, f"song_{style}_{bpm}", fmt), category="song", type="full_song", bpm=bpm,
                          key=(tonic, mode), root=None, variant=f"{style}/progression{var}"))
    return items


def load_dataset(folder: str, labels: str) -> list[dict]:
    items = []
    with open(labels, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            item = {"path": os.path.join(folder, row["file"]), "category": (row.get("category") or "loop").lower(),
                    "type": row.get("type") or None, "variant": "dataset"}
            b = (row.get("bpm") or "").strip().lower()
            item["bpm"] = "none" if b in ("none", "n/a") else float(b) if b else None
            k = (row.get("key") or "").strip()
            item["key"] = "none" if k.lower() in ("none", "n/a") else music.parse_key(k) if k else None
            r = (row.get("root") or "").strip()
            if r.lower() in ("none", "n/a"):
                item["root"] = "none"
            elif r:
                import librosa

                if r[-1].isdigit():
                    item["root"] = int(round(librosa.note_to_midi(r)))
                else:
                    item["root"], item["root_pc_only"] = int(librosa.note_to_midi(r + "4")) % 12, True
            else:
                item["root"] = None
            items.append(item)
    return items


# ------------------------------------------------------------------ analysis
def _run(item: dict) -> dict:
    warnings.filterwarnings("ignore")
    from audioscope.engine import analyze_file

    try:
        return {"item": item, "result": analyze_file(item["path"]).to_dict()}
    except Exception as e:  # noqa: BLE001
        return {"item": item, "error": str(e)}


def score(rows: list[dict]) -> dict:
    out: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        it, r = row["item"], row.get("result")
        cat = it["category"]
        if r is None:
            out[cat]["errors"].append(1)
            continue
        if it.get("type"):
            out[cat]["type"].append(r["classification"]["audio_type"] == it["type"])
        # ---- tempo
        bpm, rel = r["tempo"]["bpm"], r["tempo"]["reliability"]
        if it.get("bpm") == "none":
            out[cat]["bpm_na_correct"].append(bpm is None)
        elif isinstance(it.get("bpm"), (int, float)):
            truth = float(it["bpm"])
            exact = bpm is not None and abs(bpm - truth) <= 0.5
            acc1 = bpm is not None and abs(bpm / truth - 1) <= 0.04
            acc2 = bpm is not None and any(abs(bpm * f / truth - 1) <= 0.04 for f in (1, 2, 0.5, 3, 1 / 3))
            key = "bpm_ambiguous" if it.get("ambiguous") else "bpm"
            out[cat][f"{key}_exact"].append(exact)
            out[cat][f"{key}_acc1"].append(acc1)
            out[cat][f"{key}_acc2"].append(acc2)
            out[cat][f"{key}_na"].append(bpm is None)
            out["calibration"][f"bpm|{rel}"].append(acc1)
        # ---- key
        k = r["key"]["key"]
        if it.get("key") == "none":
            out[cat]["key_na_correct"].append(k is None)
        elif isinstance(it.get("key"), tuple):
            t_tonic, t_mode = it["key"]
            if k is None:
                s, ex = 0.0, False
            else:
                got = (k["tonic"], k["mode"])
                ex = got == (t_tonic, t_mode)
                if ex:
                    s = 1.0
                elif got[1] == t_mode and (got[0] - t_tonic) % 12 in (5, 7):
                    s = 0.5
                elif got == music.relative_key(t_tonic, t_mode):
                    s = 0.3
                elif got[0] == t_tonic:
                    s = 0.2
                else:
                    s = 0.0
            out[cat]["key_exact"].append(ex)
            out[cat]["key_mirex"].append(s)
            out[cat]["key_na"].append(k is None)
            out["calibration"][f"key|{r['key']['reliability']}"].append(ex)
        # ---- root
        root = r["pitch"]["root_midi"]
        if it.get("root") == "none":
            out[cat]["root_na_correct"].append(root is None)
        elif isinstance(it.get("root"), int):
            if it.get("root_pc_only"):
                out[cat]["root_pc"].append(root is not None and root % 12 == it["root"] % 12)
            else:
                out[cat]["root_pc"].append(root is not None and root % 12 == it["root"] % 12)
                out[cat]["root_exact"].append(root == it["root"])
                if root == it["root"] and "cents" in it and r["pitch"]["root_cents"] is not None:
                    out[cat]["root_cents_err"].append(abs(r["pitch"]["root_cents"] - it["cents"]))
            out["calibration"][f"root|{r['pitch']['reliability']}"].append(root is not None and root % 12 == it["root"] % 12)
    return out


def report(scores: dict, n_items: int, source: str) -> str:
    def pct(v):
        return f"{100 * np.mean(v):.1f}% (n={len(v)})" if v else "–"

    lines = [f"# AudioScope accuracy report", "",
             f"*{dt.datetime.now():%Y-%m-%d %H:%M}* · source: **{source}** · {n_items} files", ""]
    labels = [
        ("type", "Audio type correct"),
        ("bpm_exact", "BPM exact (±0.5)"), ("bpm_acc1", "BPM Acc1 (±4 %)"), ("bpm_acc2", "BPM Acc2 (octave errors allowed)"),
        ("bpm_na", "BPM reported N/A (should be low)"),
        ("bpm_ambiguous_exact", "BPM exact – half/double-ambiguous grooves"),
        ("bpm_ambiguous_acc2", "BPM Acc2 – half/double-ambiguous grooves"),
        ("bpm_na_correct", "BPM correctly N/A (no tempo)"),
        ("key_exact", "Key exact"), ("key_mirex", "Key MIREX weighted score"), ("key_na", "Key reported N/A"),
        ("key_na_correct", "Key correctly N/A (single note / no key)"),
        ("root_pc", "Root pitch class correct"), ("root_exact", "Root exact note (with octave)"),
        ("root_na_correct", "Root correctly N/A (noise)"), ("errors", "Decode/analysis errors"),
    ]
    for cat, title in (("song", "Full songs"), ("loop", "Loops"), ("oneshot", "One-shots")):
        if cat not in scores:
            continue
        lines += [f"## {title}", "", "| Metric | Result |", "|---|---|"]
        s = scores[cat]
        for k, label in labels:
            if s.get(k):
                if k == "key_mirex":
                    lines.append(f"| {label} | {np.mean(s[k]):.3f} (n={len(s[k])}) |")
                elif k == "errors":
                    lines.append(f"| {label} | {len(s[k])} |")
                else:
                    lines.append(f"| {label} | {pct(s[k])} |")
        if s.get("root_cents_err"):
            lines.append(f"| Root cents error (median) | {np.median(s['root_cents_err']):.1f}¢ |")
        lines.append("")
    cal = scores.get("calibration", {})
    if cal:
        lines += ["## Reliability calibration", "",
                  "Accuracy of results grouped by the reliability AudioScope reported. A well-behaved",
                  "indicator shows High > Medium > Low. (BPM: Acc1; key: exact; root: pitch class.)", "",
                  "| Detector | High | Medium | Low |", "|---|---|---|---|"]
        for det in ("bpm", "key", "root"):
            cells = [pct(cal.get(f"{det}|{lvl}", [])) for lvl in ("high", "medium", "low")]
            lines.append(f"| {det} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=0, help="number of synthetic items (plus ~n/8 songs)")
    ap.add_argument("--dataset")
    ap.add_argument("--labels")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results"))
    ap.add_argument("--failures", action="store_true", help="list every wrong result")
    args = ap.parse_args()
    if not args.synthetic and not args.dataset:
        ap.error("use --synthetic N and/or --dataset DIR --labels CSV")

    with tempfile.TemporaryDirectory(prefix="audioscope_eval_") as tmp:
        items, sources = [], []
        if args.synthetic:
            print(f"Rendering synthetic corpus ({args.synthetic} items + songs)…", flush=True)
            items += make_synthetic(args.synthetic, tmp, args.seed)
            sources.append(f"synthetic (seed {args.seed})")
        if args.dataset:
            items += load_dataset(args.dataset, args.labels)
            sources.append(f"dataset {args.dataset}")
        print(f"Analysing {len(items)} files with {args.workers} workers…", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_run, items))
        scores = score(rows)
        text = report(scores, len(items), " + ".join(sources))
        if args.failures:
            text += "\n## Failures\n\n" + "\n".join(_failures(rows)) + "\n"
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"report_{dt.datetime.now():%Y%m%d_%H%M%S}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved {out}")


def _failures(rows: list[dict]) -> list[str]:
    out = []
    for row in rows:
        it, r = row["item"], row.get("result")
        name = os.path.basename(it["path"])
        if r is None:
            out.append(f"- {name}: ERROR {row.get('error')}")
            continue
        bpm = r["tempo"]["bpm"]
        if isinstance(it.get("bpm"), (int, float)) and (bpm is None or abs(bpm - it["bpm"]) > 0.5):
            out.append(f"- {name} [{it['variant']}]: BPM {bpm} vs {it['bpm']} ({r['tempo']['reliability']})")
        k = r["key"]["key"]
        if isinstance(it.get("key"), tuple) and (k is None or (k["tonic"], k["mode"]) != it["key"]):
            got = music.key_name(k["tonic"], k["mode"]) if k else "N/A"
            out.append(f"- {name} [{it['variant']}]: key {got} vs {music.key_name(*it['key'])} ({r['key']['reliability']})")
        root = r["pitch"]["root_midi"]
        if isinstance(it.get("root"), int) and (root is None or root % 12 != it["root"] % 12):
            out.append(f"- {name} [{it['variant']}]: root {root} vs {it['root']}")
        if it.get("type") and r["classification"]["audio_type"] != it["type"]:
            out.append(f"- {name}: type {r['classification']['audio_type']} vs {it['type']}")
    return out


if __name__ == "__main__":
    main()
