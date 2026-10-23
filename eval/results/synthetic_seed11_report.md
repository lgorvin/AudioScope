# AudioScope accuracy report

*2026-10-23 21:06* · source: **synthetic (seed 11)** · 180 files

## Full songs

| Metric | Result |
|---|---|
| Audio type correct | 100.0% (n=20) |
| BPM exact (±0.5) | 95.0% (n=20) |
| BPM Acc1 (±4 %) | 95.0% (n=20) |
| BPM Acc2 (octave errors allowed) | 100.0% (n=20) |
| BPM reported N/A (should be low) | 0.0% (n=20) |
| Key exact | 80.0% (n=20) |
| Key MIREX weighted score | 0.860 (n=20) |
| Key reported N/A | 0.0% (n=20) |

## Loops

| Metric | Result |
|---|---|
| Audio type correct | 96.8% (n=125) |
| BPM exact (±0.5) | 85.5% (n=110) |
| BPM Acc1 (±4 %) | 85.5% (n=110) |
| BPM Acc2 (octave errors allowed) | 93.6% (n=110) |
| BPM reported N/A (should be low) | 6.4% (n=110) |
| BPM exact – half/double-ambiguous grooves | 70.0% (n=10) |
| BPM Acc2 – half/double-ambiguous grooves | 90.0% (n=10) |
| Key exact | 83.5% (n=85) |
| Key MIREX weighted score | 0.891 (n=85) |
| Key reported N/A | 0.0% (n=85) |

## One-shots

| Metric | Result |
|---|---|
| Audio type correct | 100.0% (n=35) |
| BPM correctly N/A (no tempo) | 100.0% (n=35) |
| Key correctly N/A (single note / no key) | 100.0% (n=29) |
| Root pitch class correct | 100.0% (n=25) |
| Root exact note (with octave) | 100.0% (n=19) |
| Root correctly N/A (noise) | 100.0% (n=5) |
| Root cents error (median) | 2.2¢ |

## Reliability calibration

Accuracy of results grouped by the reliability AudioScope reported. A well-behaved
indicator shows High > Medium > Low. (BPM: Acc1; key: exact; root: pitch class.)

| Detector | High | Medium | Low |
|---|---|---|---|
| bpm | 98.7% (n=78) | 79.6% (n=54) | – |
| key | 95.8% (n=24) | 78.7% (n=61) | 80.0% (n=20) |
| root | 100.0% (n=19) | 100.0% (n=6) | – |
