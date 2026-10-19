# Nutate

Bloch simulations of any pulse sequence, rendered as 3D Bloch-sphere videos.

![Chirp excitation on the Bloch sphere](https://raw.githubusercontent.com/mforoozandeh/nutate/main/docs/media/chirp.gif)

Nutate takes any pulse, given as a waveform (amplitude and phase, or I and Q) with its
duration and Rabi frequency, whether it drives nuclear spins (NMR), electron spins (EPR/ESR)
or a qubit. It simulates the Bloch vector in the rotating frame for many offsets at once,
renders the trajectories with [PyVista](https://pyvista.org) as MP4/GIF videos, PNG stills
or an interactive viewer, and maps robustness to drive (B1) amplitude errors. Optional pulse
families (chirps, hyperbolic-secant pulses, CHORUS, composite pulses) live in
`nutate.pulses`.

## Install

```bash
pip install nutate
```

Python ≥ 3.11. Video encoding uses the ffmpeg binary bundled with `imageio-ffmpeg`; no
system ffmpeg or LaTeX needed.

## Quickstart

A generator makes the pulse: a waveform normalised to 1 plus its duration and Rabi frequency
ν1 (the drive amplitude, RF or MW, at waveform value 1). `nt.run` simulates it and writes
the video and/or the B1 robustness figures.

```python
import nutate as nt
from nutate.pulses.swept_frequency import chirp

pulse = chirp(10e-3, 20e3, 500, smoothing=10)  # pulse.amp (peak 1), .phase, .duration, .rabi
nt.run(pulse, 10e3, still=True, b1=True, out="output", name="chirp")
# output/chirp.mp4, chirp.png, chirp_b1_map.png, chirp_b1_surface.png, chirp_b1_phase.png
```

Units are SI throughout (seconds, hertz, radians); arguments in degrees end in `_deg`.

The package core holds the pulse type, `nt.waveform` (your own arrays) and `nt.delay` (free
precession). Pulse families live in `nutate.pulses`:

- `nutate.pulses.swept_frequency`: `chirp`, `hs`, `chorus` (a list including its delay),
  `CHORUS_CYCLE`, `phase_correction` and `chorus_phase_correction`
- `nutate.pulses.composite`: `composite` (a list, one pulse per element) and `PRESETS`

`pulse.with_rabi(ν1)` changes the Rabi frequency and keeps the waveform.

### Your own waveform

`nt.waveform` turns arrays from any other generator (an optimiser, a shape tool) into a
pulse: amplitude and phase, or I and Q, normalised to 1. `rabi` is the Rabi frequency in
Hz, or give `t90`, the hard 90° pulse length (`rabi = 1/(4·t90)`). Slight overshoot above 1
is simulated as given; a peak above 1.5 is an error.

```python
p1 = nt.waveform(amp, phase, duration=2e-3, rabi=15e3)  # or phase_deg=...
p2 = nt.waveform(i=I, q=Q, duration=1e-3, t90=10e-6)
result = nt.run([p1, p2], 300e3, b1=True, out="output", name="my_pulse")
```

### Sequences and delays

A sequence is a list, played back to back. Put `nt.delay(duration, n_steps)` between
pulses for free precession (no drive); `n_steps` sets how finely the delay is sampled, i.e.
how smoothly the video shows it.

```python
sequence = [p1, nt.delay(500e-6, 100), p2]  # p1, 500 µs of free precession, then p2
nt.run(sequence, 300e3, out="output", name="echo")
```

### `nt.run` options

- Offsets span `±bandwidth/2`: `n_offsets` (31) for the video, `b1_offsets` (101) for the
  B1 map. Or pass `offsets=` (Hz) for both.
- `video` (on), `still` (last frame as PNG), `b1` (off), `show` (interactive windows
  instead of files). Pulses in a list play back to back; `phase_cycle=` is passed through.
- `video_options` go to `nt.render`; those that apply also go to the still and the viewer:
  - `offsets`: draw only `k` evenly spaced offsets, or given indices
  - `mode`: `"arrows"`, `"ribbon"` (a curve through all tips) or `"both"`
  - `trail`: fading tip trails, as a fraction of the sequence
  - `inset` / `inset_phase`: drive amplitude (and phase) vs time with a moving cursor
  - `theme`: `"light"` (default) or `"dark"`
  - `camera=nt.Camera(orbit=90)`, `style=nt.Style(...)` (colours, sizes, font), `duration`,
    `size`
- B1 figures plot `−My` by default; `component="z"` for inversion. `b1_scales` (0.5–1.5) and
  `rabi_ref` (the highest ν1, for the `Δ/Ω1(0)` axis) can be changed.
- The result holds the trajectory, the B1 map, the phase sensitivity and the files written.

### Lower level

```python
import numpy as np

traj = nt.simulate([pulse], offsets=np.linspace(-5e3, 5e3, 31))
nt.render(traj, "chirp.mp4", duration=12)
```

`nt.snapshot(...)` writes a PNG still; `nt.show(traj)` opens an interactive window with a
time slider. `nt.plot_profiles(traj)` and `nt.plot_map(traj)` make the 2D excitation
profile and time–offset map plots.

### B1 robustness

```python
pulse = nt.waveform([1.0], [0.0], duration=10e-6, t90=10e-6)  # hard 90°, ν1 = 25 kHz
offsets = np.linspace(-100e3, 100e3, 101)  # Δ/Ω1(0) from −4 to 4
scales = np.linspace(0.5, 1.5, 101)  # Ω1 / Ω1(0)
bmap = nt.b1_map([pulse], offsets, scales)
nt.plot_b1_map(bmap, "y", sign=-1)  # contours over Δ/Ω1(0) × Ω1/Ω1(0)
nt.plot_b1_map(bmap, "y", sign=-1, surface=True)
nt.plot_b1_phase(scales, nt.b1_phase_sensitivity([pulse], scales))
```

Sequences whose phase varies quickly across the band, such as a single chirp, need dense
offsets; `examples/b1_chirp.py` uses 1000, as the original script did.

`simulate(..., b1=...)` scales every drive amplitude (one factor, or one per offset);
`simulate_final` returns only the final states, which keeps large grids small in memory.
`nutate.pulses.swept_frequency.phase_correction(sequence, bandwidth, ...)` fits the
first-chirp phase correction for any CHORUS-like sequence.

## Examples

Each script generates its pulses and passes them to `nt.run`. Video scripts write
`output/<name>.mp4` and `.png`; the `b1_*` scripts (and `waveform.py`) write
`output/<name>_b1_map.png`, `_b1_surface.png` and `_b1_phase.png`. To change what they
produce, edit the `nt.run` call: `show=True` for interactive windows, `out=` for another
folder, `video_options={"theme": "dark"}` for a dark video.

| Example | Shows |
|---|---|
| [`examples/chirp.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/chirp.py) | linear chirp excitation |
| [`examples/chirp_rewind.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/chirp_rewind.py) | a chirp, then an ideal rewind of its quadratic phase |
| [`examples/hard_pulse.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/hard_pulse.py) | a hard 90° pulse over a wide offset range |
| [`examples/composite.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/composite.py) | a composite pulse (preset 9 on a 10 µs 90° pulse) |
| [`examples/chorus.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/chorus.py) | CHORUS with phase correction, plus excitation profiles |
| [`examples/compressed_chorus_epr.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/compressed_chorus_epr.py) | compressed CHORUS for EPR, plus a time–offset map and phase paths |
| [`examples/waveform.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/waveform.py) | your own I/Q arrays (a shifted Gaussian 90°): video, still and B1 figures |
| [`examples/b1_hard.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/b1_hard.py) | B1 map and phase sensitivity: hard 90° pulse |
| [`examples/b1_composite.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/b1_composite.py) | B1 map and phase sensitivity: composite pulses |
| [`examples/b1_chirp.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/b1_chirp.py) | B1 map and phase sensitivity: single chirp |
| [`examples/b1_double_chirp.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/b1_double_chirp.py) | B1 map and phase sensitivity: double chirp |
| [`examples/b1_chorus.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/b1_chorus.py) | B1 map and phase sensitivity: CHORUS |
| [`examples/b1_abstruse.py`](https://github.com/mforoozandeh/nutate/blob/main/examples/b1_abstruse.py) | B1 map and phase sensitivity: ABSTRUSE (hyperbolic-secant pulses) |

## Conventions

- Rotating frame, no relaxation; initial magnetization `(0, 0, 1)`.
- Each step rotates the magnetization by `+α = 2π·sqrt(Ω² + Δ²)·dt` (right-handed) about
  `n = (sinθ cosφ, sinθ sinφ, cosθ)` with `θ = atan2(Ω, Δ)`, for drive amplitude `Ω`
  (waveform × ν1), phase `φ` and offset `Δ` — i.e. `Rz(φ)·Ry(θ)·Rz(α)·Ry(−θ)·Rz(−φ)`. So an
  on-resonance 90° pulse with phase 0 takes `z → −y`.
- Profile phase is `angle(My + i·Mx)`.

## Headless rendering

Videos are rendered off-screen. On a server without a display, VTK needs an off-screen
OpenGL context (OSMesa/EGL or a virtual display); see the off-screen rendering notes in the
[PyVista documentation](https://docs.pyvista.org).

## Citation

If you use `nutate` in published work, please cite it; see [`CITATION.cff`](https://github.com/mforoozandeh/nutate/blob/main/CITATION.cff).

## License

Apache License 2.0 — see [`LICENSE`](https://github.com/mforoozandeh/nutate/blob/main/LICENSE) and [`NOTICE`](https://github.com/mforoozandeh/nutate/blob/main/NOTICE).
