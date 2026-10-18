import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

import nutate as nt

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.render
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(script, tmp_path, monkeypatch):
    argv = []
    if script.stem == "bruker_shape":
        shape = tmp_path / "shape.txt"
        shape.write_text("##XYPOINTS= (XY..XY)\n" + "100.0, 0.0\n" * 50 + "##END=\n")
        argv = [str(shape), "1e-3", "250"]
    real_run = nt.run

    def quick_run(*args, **kwargs):  # tiny, short renders and an 11 × 11 B1 grid
        video_options = kwargs.pop("video_options", None) or {}
        kwargs.update(b1_offsets=11, b1_scales=np.linspace(0.5, 1.5, 11))
        video_options = {**video_options, "duration": 0.5, "hold": 0, "size": (320, 240)}
        return real_run(*args, video_options=video_options, **kwargs)

    monkeypatch.setattr(nt, "run", quick_run)
    monkeypatch.setattr(sys, "argv", [script.name, *argv])
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")

    out = tmp_path / "output"
    if script.stem.startswith("b1_") or script.stem == "waveform":
        for suffix in ("map", "surface", "phase"):
            assert (out / f"{script.stem}_b1_{suffix}.png").exists()
    if not script.stem.startswith("b1_"):  # B1 robustness examples write no video
        assert (out / f"{script.stem}.mp4").exists()
        assert (out / f"{script.stem}.png").exists()


def test_all_examples_present():
    assert {p.stem for p in EXAMPLES} == {
        "b1_abstruse",
        "b1_chirp",
        "b1_chorus",
        "b1_composite",
        "b1_double_chirp",
        "b1_hard",
        "bruker_shape",
        "chirp",
        "chirp_rewind",
        "chorus",
        "composite",
        "compressed_chorus_epr",
        "hard_pulse",
        "waveform",
    }


def test_epr_phase_paths_skip_receiver_phase(monkeypatch):
    """All pulse phases advanced by 90°; the receiver phase is NOT applied to the path."""
    import importlib
    from dataclasses import replace

    import numpy as np

    import nutate as nt

    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    example = importlib.import_module("compressed_chorus_epr")
    sequence = nt.chorus(200e-9, 100e-9, 200e6, 0.625e-9)
    offsets = np.array([-10e6, 0.0, 10e6])
    t, paths = example.phase_paths(sequence, offsets)
    shifted = [replace(s, phase=s.phase + np.pi / 2) for s in sequence]
    m = nt.simulate(shifted, offsets).m
    np.testing.assert_allclose(paths, np.arctan2(m[:, :, 1], m[:, :, 0]) / np.pi, atol=1e-12)
    assert t.shape == (m.shape[0],)


def _captured_run(monkeypatch, tmp_path, stem, *argv):
    """Run examples/<stem>.py with nt.run captured; returns its arguments, defaults filled in."""
    import importlib
    import inspect

    import nutate as nt

    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    example = importlib.import_module(stem)
    seen = {}
    signature = inspect.signature(importlib.import_module("nutate.run").run)  # unpatched

    def fake_run(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.update(bound.arguments)
        seen["sequence"] = (
            [seen["sequence"]] if isinstance(seen["sequence"], nt.Segment) else seen["sequence"]
        )
        return nt.RunResult(None, None, None, [])

    monkeypatch.setattr(nt, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [stem, *argv])
    example.main()
    return example, seen


def _peaks(sequence):  # Hz
    return [s.rf.max() for s in sequence]


def _rf_ref(run):
    return run["rf_ref"] if run["rf_ref"] is not None else max(s.rabi for s in run["sequence"])


def test_b1_examples_pin_constants(monkeypatch, tmp_path):
    import numpy as np
    import pytest as _pytest

    import nutate as nt

    approx = _pytest.approx

    ex, run = _captured_run(monkeypatch, tmp_path, "b1_hard")
    assert run["b1"] and not run["video"]
    assert run["bandwidth"] == ex.BANDWIDTH and run["b1_offsets"] == 101 and run["sign"] == -1
    assert _rf_ref(run) == approx(1 / (4 * ex.T))
    assert run["phase_cycle"].receiver_deg.tolist() == [0, 90, 180, 270]

    ex, run = _captured_run(monkeypatch, tmp_path, "b1_chirp")
    q5 = np.sqrt(ex.T * ex.BANDWIDTH * 5 / (2 * np.pi)) / ex.T
    assert run["b1_offsets"] == 1000  # dense: ~120 phase cycles across the band
    assert run["sign"] == -1 and _rf_ref(run) == approx(q5)
    assert _peaks(run["sequence"]) == approx([q5 / 3.3588])

    ex, run = _captured_run(monkeypatch, tmp_path, "b1_double_chirp")
    q5 = np.sqrt(ex.T / 2 * ex.BANDWIDTH * 5 / (2 * np.pi)) / (ex.T / 2)
    assert run["sign"] == 1 and _rf_ref(run) == approx(q5)
    assert _peaks(run["sequence"]) == approx([q5 / 4.75, q5, 0.0])
    assert [s.duration for s in run["sequence"]] == approx([ex.T, ex.T / 2, ex.T / 2])
    assert run["phase_cycle"].pulse_phases_deg[:, 1].tolist() == [0, 90, 180, 270]
    assert run["phase_cycle"].receiver_deg.tolist() == [0, 180, 0, 180]

    ex, run = _captured_run(monkeypatch, tmp_path, "b1_chorus")
    q5 = np.sqrt(ex.T / 2 * ex.BANDWIDTH * 5 / (2 * np.pi)) / (ex.T / 2)
    assert run["sign"] == -1 and _rf_ref(run) == approx(q5)
    assert _peaks(run["sequence"]) == approx([q5 / 4.75, 0.9661 * q5 / np.sqrt(2), 0.0, q5])
    assert [s.duration for s in run["sequence"]] == approx([ex.T, ex.T, ex.T / 2, ex.T / 2])
    assert run["phase_cycle"] is nt.CHORUS_CYCLE

    ex, run = _captured_run(monkeypatch, tmp_path, "b1_abstruse")
    k = [0.61 * np.sqrt(ex.BANDWIDTH / ex.T), 1.83 * np.sqrt(ex.BANDWIDTH / ex.T)]
    k3 = 1.93 * np.sqrt(ex.BANDWIDTH / (ex.T / 2))
    assert run["sign"] == 1 and _rf_ref(run) == approx(k3)
    assert _peaks(run["sequence"]) == approx([*k, 0.0, k3], rel=1e-4)  # sech peak off-grid
    assert run["phase_cycle"] is nt.CHORUS_CYCLE


def test_b1_composite_presets(monkeypatch, tmp_path):
    import numpy as np

    _, run = _captured_run(monkeypatch, tmp_path, "b1_composite", "--preset", "14")
    assert run["sign"] == -1
    assert len(run["sequence"]) == 48  # preset 14: 48 × 90°
    assert all(s.n_steps == 90 for s in run["sequence"])
    _, run = _captured_run(monkeypatch, tmp_path, "b1_composite", "--preset", "12")
    np.testing.assert_allclose([s.phase[0] for s in run["sequence"]], np.deg2rad([90, 0, 90]))
    _, run = _captured_run(monkeypatch, tmp_path, "b1_composite", "--preset", "10")
    assert [s.n_steps for s in run["sequence"]] == [180, 360, 180, 90]


def test_composite_examples_offset_window_in_rf_units(monkeypatch, tmp_path):
    import pytest as _pytest

    rf = 1 / (4 * 10e-6)  # Ω1 for t90 = 10 µs
    for argv, window in (([], 0.8), (["--window", "2"], 2.0)):
        _, run = _captured_run(monkeypatch, tmp_path, "composite", *argv)
        assert run["bandwidth"] == _pytest.approx(2 * window * rf)
        assert run["n_offsets"] == 31

    _, run = _captured_run(monkeypatch, tmp_path, "b1_composite")
    assert run["bandwidth"] == _pytest.approx(1.6 * rf) and run["b1_offsets"] == 101
    _, run = _captured_run(monkeypatch, tmp_path, "b1_composite", "--window", "3")
    assert run["bandwidth"] == _pytest.approx(6 * rf)
