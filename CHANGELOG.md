# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-19

First release.

### Added

- Rotating-frame Bloch simulation of any pulse sequence for many offsets at once, stepped
  with the exact SU(2) propagator; phase cycling and drive (B1) amplitude scaling.
- Pulses as a waveform normalised to 1 plus its Rabi frequency: `nt.waveform` from
  amplitude and phase or I and Q, with `rabi` or the hard 90° length `t90`; `nt.delay` for
  free precession.
- `nt.run`: one call from a pulse or sequence to a Bloch-sphere video, a still, B1
  robustness figures (offset × B1 map, surface, phase sensitivity) or interactive windows.
- Rendering with PyVista to MP4/GIF and PNG or an interactive viewer: arrows or ribbons,
  fading trails, drive amplitude and phase insets, light and dark themes, orbiting camera.
- Optional pulse families: `nutate.pulses.swept_frequency` (chirps, hyperbolic-secant
  pulses, CHORUS, phase correction) and `nutate.pulses.composite` (composite pulses and
  presets).

[0.1.0]: https://github.com/mforoozandeh/nutate/releases/tag/v0.1.0
