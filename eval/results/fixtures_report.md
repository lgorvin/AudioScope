# AudioScope accuracy report

*2026-10-23 21:06* · source: **dataset tests/fixtures** · 16 files

## Full songs

| Metric | Result |
|---|---|
| Audio type correct | 100.0% (n=2) |
| BPM exact (±0.5) | 100.0% (n=2) |
| BPM Acc1 (±4 %) | 100.0% (n=2) |
| BPM Acc2 (octave errors allowed) | 100.0% (n=2) |
| BPM reported N/A (should be low) | 0.0% (n=2) |
| Key exact | 100.0% (n=2) |
| Key MIREX weighted score | 1.000 (n=2) |
| Key reported N/A | 0.0% (n=2) |

## Loops

| Metric | Result |
|---|---|
| Audio type correct | 100.0% (n=9) |
| BPM exact (±0.5) | 100.0% (n=7) |
| BPM Acc1 (±4 %) | 100.0% (n=7) |
| BPM Acc2 (octave errors allowed) | 100.0% (n=7) |
| BPM reported N/A (should be low) | 0.0% (n=7) |
| BPM correctly N/A (no tempo) | 100.0% (n=2) |
| Key exact | 100.0% (n=6) |
| Key MIREX weighted score | 1.000 (n=6) |
| Key reported N/A | 0.0% (n=6) |

## One-shots

| Metric | Result |
|---|---|
| Audio type correct | 100.0% (n=5) |
| BPM correctly N/A (no tempo) | 100.0% (n=5) |
| Key correctly N/A (single note / no key) | 100.0% (n=4) |
| Root pitch class correct | 100.0% (n=4) |
| Root exact note (with octave) | 100.0% (n=3) |
| Root correctly N/A (noise) | 100.0% (n=1) |

## Reliability calibration

Accuracy of results grouped by the reliability AudioScope reported. A well-behaved
indicator shows High > Medium > Low. (BPM: Acc1; key: exact; root: pitch class.)

| Detector | High | Medium | Low |
|---|---|---|---|
| bpm | 100.0% (n=8) | 100.0% (n=1) | – |
| key | 100.0% (n=4) | 100.0% (n=3) | 100.0% (n=1) |
| root | 100.0% (n=3) | 100.0% (n=1) | – |
