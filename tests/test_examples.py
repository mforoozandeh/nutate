import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLES = sorted(p for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_"))


@pytest.mark.render
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(script, tmp_path):
    extra = []
    if script.stem == "bruker_shape":
        shape = tmp_path / "shape.txt"
        shape.write_text("##XYPOINTS= (XY..XY)\n" + "100.0, 0.0\n" * 50 + "##END=\n")
        extra = [str(shape), "1e-3", "250"]
    subprocess.run(
        [sys.executable, str(script), *extra, "--quick", "--out", str(tmp_path)],
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if script.stem.startswith("b1_"):  # B1 robustness examples write figures, not videos
        for suffix in ("map", "surface", "phase"):
            assert (tmp_path / f"{script.stem}_{suffix}.png").exists()
    else:
        assert (tmp_path / f"{script.stem}.mp4").exists()


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
    }


def test_epr_phase_paths_skip_receiver_phase(monkeypatch):
    """All pulse phases advanced by 90°; the receiver phase is NOT applied to the path."""
    import importlib

    import numpy as np

    import nutate as nt

    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    example = importlib.import_module("compressed_chorus_epr")
    sequence = nt.chorus(200e-9, 100e-9, 200e6, 0.625e-9)
    offsets = np.array([-10e6, 0.0, 10e6])
    t, paths = example.phase_paths(sequence, offsets)
    shifted = [nt.Segment(s.name, s.amp, s.phase + np.pi / 2, s.dt) for s in sequence]
    m = nt.simulate(shifted, offsets).m
    np.testing.assert_allclose(paths, np.arctan2(m[:, :, 1], m[:, :, 0]) / np.pi, atol=1e-12)
    assert t.shape == (m.shape[0],)


def _b1_run(monkeypatch, tmp_path, stem, *argv):
    """Run examples/<stem>.py with finish_b1 captured; returns its arguments."""
    import importlib

    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    example = importlib.import_module(stem)
    seen = {}
    monkeypatch.setattr(
        example,
        "finish_b1",
        lambda name, args, sequence, offsets, scales, **kw: seen.update(
            name=name, sequence=sequence, offsets=offsets, scales=scales, **kw
        ),
    )
    monkeypatch.setattr(sys, "argv", [stem, *argv, "--out", str(tmp_path)])
    example.main()
    return example, seen


def _peaks(sequence):
    return [s.amp.max() for s in sequence]


def test_b1_examples_pin_constants(monkeypatch, tmp_path):
    import numpy as np
    import pytest as _pytest

    import nutate as nt

    approx = _pytest.approx

    ex, run = _b1_run(monkeypatch, tmp_path, "b1_hard")
    assert len(run["offsets"]) == 101 and run["sign"] == -1
    assert run["rf_ref"] == approx(1 / (4 * ex.T))
    assert run["phase_cycle"].receiver_deg.tolist() == [0, 90, 180, 270]

    ex, run = _b1_run(monkeypatch, tmp_path, "b1_chirp")
    q5 = np.sqrt(ex.T * ex.BANDWIDTH * 5 / (2 * np.pi)) / ex.T
    assert len(run["offsets"]) == 1000  # dense: ~120 phase cycles across the band
    assert run["sign"] == -1 and run["rf_ref"] == approx(q5)
    assert _peaks(run["sequence"]) == approx([q5 / 3.3588])

    ex, run = _b1_run(monkeypatch, tmp_path, "b1_double_chirp")
    q5 = np.sqrt(ex.T / 2 * ex.BANDWIDTH * 5 / (2 * np.pi)) / (ex.T / 2)
    assert run["sign"] == 1 and run["rf_ref"] == approx(q5)
    assert _peaks(run["sequence"]) == approx([q5 / 4.75, q5, 0.0])
    assert [s.duration for s in run["sequence"]] == approx([ex.T, ex.T / 2, ex.T / 2])
    assert run["phase_cycle"].pulse_phases_deg[:, 1].tolist() == [0, 90, 180, 270]
    assert run["phase_cycle"].receiver_deg.tolist() == [0, 180, 0, 180]

    ex, run = _b1_run(monkeypatch, tmp_path, "b1_chorus")
    q5 = np.sqrt(ex.T / 2 * ex.BANDWIDTH * 5 / (2 * np.pi)) / (ex.T / 2)
    assert run["sign"] == -1 and run["rf_ref"] == approx(q5)
    assert _peaks(run["sequence"]) == approx([q5 / 4.75, 0.9661 * q5 / np.sqrt(2), 0.0, q5])
    assert [s.duration for s in run["sequence"]] == approx([ex.T, ex.T, ex.T / 2, ex.T / 2])
    assert run["phase_cycle"] is nt.CHORUS_CYCLE

    ex, run = _b1_run(monkeypatch, tmp_path, "b1_abstruse")
    k = [0.61 * np.sqrt(ex.BANDWIDTH / ex.T), 1.83 * np.sqrt(ex.BANDWIDTH / ex.T)]
    k3 = 1.93 * np.sqrt(ex.BANDWIDTH / (ex.T / 2))
    assert run["sign"] == 1 and run["rf_ref"] == approx(k3)
    assert _peaks(run["sequence"]) == approx([*k, 0.0, k3], rel=1e-4)  # sech peak off-grid
    assert run["phase_cycle"] is nt.CHORUS_CYCLE


def test_b1_composite_presets(monkeypatch, tmp_path):
    import numpy as np

    _, run = _b1_run(monkeypatch, tmp_path, "b1_composite", "--preset", "14")
    assert run["sign"] == -1
    assert len(run["sequence"]) == 48  # preset 14: 48 × 90°
    assert all(s.n_steps == 90 for s in run["sequence"])
    _, run = _b1_run(monkeypatch, tmp_path, "b1_composite", "--preset", "12")
    np.testing.assert_allclose([s.phase[0] for s in run["sequence"]], np.deg2rad([90, 0, 90]))
    _, run = _b1_run(monkeypatch, tmp_path, "b1_composite", "--preset", "10")
    assert [s.n_steps for s in run["sequence"]] == [180, 360, 180, 90]


def test_composite_examples_offset_window_in_rf_units(monkeypatch, tmp_path):
    import importlib

    import numpy as np

    rf = 1 / (4 * 10e-6)  # Ω1 for t90 = 10 µs
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    composite = importlib.import_module("composite")
    seen = {}
    monkeypatch.setattr(composite, "finish", lambda traj, name, args, **kw: seen.update(traj=traj))
    for argv, window in (([], 0.8), (["--window", "2"], 2.0)):
        monkeypatch.setattr(sys, "argv", ["composite", *argv, "--out", str(tmp_path)])
        composite.main()
        np.testing.assert_allclose(seen["traj"].offsets, np.linspace(-window, window, 31) * rf)

    _, run = _b1_run(monkeypatch, tmp_path, "b1_composite")
    np.testing.assert_allclose(run["offsets"], np.linspace(-0.8, 0.8, 101) * rf)
    _, run = _b1_run(monkeypatch, tmp_path, "b1_composite", "--window", "3")
    np.testing.assert_allclose(run["offsets"][[0, -1]], [-3 * rf, 3 * rf])
