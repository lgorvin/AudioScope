# Evaluation

`evaluate.py` measures accuracy **separately for full songs, loops and one-shots**, and checks
whether the reliability labels mean something (calibration).

```powershell
# synthetic corpus (seeded; ~2-3 min for 160 items + 20 songs on 6 workers)
.venv\Scripts\python eval\evaluate.py --synthetic 160 --seed 11 --workers 6 --failures

# your own labelled audio
.venv\Scripts\python eval\evaluate.py --dataset D:\refs --labels D:\refs\labels.csv
```

Reports are written to `eval/results/`.

## Labels CSV

```csv
file,category,type,bpm,key,root
Loops/keys_01.wav,loop,melodic_loop,128,A minor,
OneShots/808_F.wav,oneshot,one_shot,none,none,F1
Songs/track.mp3,song,full_song,124,8A,
```

* `category`: `song`, `loop` or `oneshot` (controls which table the file is scored in)
* `bpm`: number, or `none` when the file has no tempo (one-shots, free-time phrases)
* `key`: `A minor`, `F#m`, `Eb`, Camelot `8A`, or `none` (single notes, drums)
* `root`: `F#2` (exact note) or `C` (pitch class only), or `none` (noise, hats)
* any column may be left empty to skip that check

## Current results (synthetic)

See [`results/synthetic_seed11_report.md`](results/synthetic_seed11_report.md).

| | Full songs | Loops | One-shots |
|---|---|---|---|
| Audio type | 100 % | 96.8 % | 100 % |
| BPM exact (±0.5) | 95 % | 85.5 % | – (correctly N/A: 100 %) |
| BPM with half/double allowed | 100 % | 93.6 % | – |
| Key exact | 80 % | 83.5 % | – (correctly N/A: 100 %) |
| Key MIREX score | 0.86 | 0.89 | – |
| Root note (exact octave) | – | – | 100 %, median error 2.2 ¢ |

Calibration (accuracy of results by the reliability AudioScope reported):

| | High | Medium | Low |
|---|---|---|---|
| BPM | 98.7 % | 79.6 % | – |
| Key | 95.8 % | 78.7 % | 80.0 % |

**These numbers are an upper bound.** The corpus is rendered by `tests/synth.py`
(synthesised drums, chords, bass, 808s, vocal-like lines), with MP3 round-trips, noise and
detuning, but it is not real music. Real recordings (dense mixes, live tempo drift, modal
or chromatic harmony, distorted 808s) will score lower. Run the dataset mode on audio you
have labelled yourself before relying on any figure, and re-check the calibration table:
if "High" is not clearly more accurate than "Medium" on your material, the thresholds in
`audioscope/engine/tempo.py` and `key.py` should be re-tuned.

## Design decisions backed by measurements

* **Pitch-class profile**: plain constant-Q chroma with square-root compression beat
  harmonic-salience and log-compressed variants for key finding (~80 % vs 70-72 % on the
  chord/bass/song benchmark). The overtone-suppressed salience profile is still used to
  *count* how many pitch classes are present (single note / power chord → key N/A).
* **Sub-bass excluded from key chroma** (below C2): kicks and 808 decay dominate that octave;
  excluding it improved bass-loop key accuracy (26 → 30 of 48) with no loss elsewhere.
* **Octave (half/double-time) choice**: autocorrelation salience alone favours the slower
  reading; adding an onset-density preference (~2 events per beat) raised octave-choice
  accuracy on synthetic grooves from 76 % to 89 %.
* **"High" key reliability** requires the relative major/minor to trail by ≥ 0.25 in
  correlation: at that threshold High-rated keys were 98 % correct on the benchmark.
