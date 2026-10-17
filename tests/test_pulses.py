import numpy as np
import pytest

from nutate import (
    COMPOSITE_PRESETS,
    Segment,
    chirp,
    composite,
    delay,
    hard,
    hs,
    q_factor,
    read_bruker_shape,
    simulate,
)


def test_segment_properties():
    s = Segment("x", amp=np.full(4, 10.0), phase=np.zeros(4), dt=1e-6)
    assert s.n_steps == 4
    assert s.duration == pytest.approx(4e-6)


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"amp": np.ones(3), "phase": np.zeros(4), "dt": 1e-6}, "phase"),
        ({"amp": -np.ones(3), "phase": np.zeros(3), "dt": 1e-6}, "amp"),
        ({"amp": np.ones(3), "phase": np.zeros(3), "dt": 0.0}, "dt"),
        ({"amp": np.array([1.0, np.nan]), "phase": np.zeros(2), "dt": 1e-6}, "amp"),
        ({"amp": np.ones(0), "phase": np.zeros(0), "dt": 1e-6}, "amp"),
    ],
)
def test_segment_validation(kwargs, field):
    with pytest.raises(ValueError, match=field):
        Segment("x", **kwargs)


def test_hard_amplitude_and_phase():
    s = hard(10e-6, flip_deg=90, phase_deg=90, n_steps=1000)
    assert s.n_steps == 1000
    assert s.duration == pytest.approx(10e-6)
    np.testing.assert_allclose(s.amp, 25e3)  # 90 / (360 · 10 µs)
    np.testing.assert_allclose(s.phase, np.pi / 2)


def test_delay_is_zero_amplitude():
    s = delay(1e-3, 50)
    assert s.n_steps == 50
    assert s.dt == pytest.approx(2e-5)
    assert not s.amp.any()


def test_q_factor_90():
    assert q_factor(90) == pytest.approx(-2 * np.log(0.5) / np.pi, rel=1e-12)  # ≈ 0.441271


@pytest.mark.parametrize("flip", [0, 180, 200])
def test_q_factor_invalid(flip):
    with pytest.raises(ValueError, match="q=5"):
        q_factor(flip)


def test_chirp_envelope():
    c = chirp(10e-3, 20e3, 500, smoothing=10)
    peak = np.sqrt(10e-3 * 20e3 * q_factor(90) / (2 * np.pi)) / 10e-3
    assert c.n_steps == 500
    assert c.dt == pytest.approx(2e-5)
    ramp = peak * np.sin(np.linspace(0, np.pi / 2, 50))
    np.testing.assert_allclose(c.amp[:50], ramp)
    np.testing.assert_allclose(c.amp[50:450], peak)
    np.testing.assert_allclose(c.amp[450:], ramp[::-1])


def test_chirp_phase_quadratic():
    c = chirp(10e-3, 20e3, 500)
    t = np.linspace(0, 10e-3, 500)
    expected = 2 * np.pi * (20e3 / t[-1] * t**2 / 2 - 10e3 * t)
    np.testing.assert_allclose(c.phase, expected - expected.min(), atol=1e-9)


def test_chirp_phase_correction_added():
    corr = np.linspace(0, 90, 500)
    np.testing.assert_allclose(
        chirp(10e-3, 20e3, 500, phase_correction_deg=corr).phase - chirp(10e-3, 20e3, 500).phase,
        np.deg2rad(corr),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"smoothing": 50}, {"smoothing": -1}, {"phase_correction_deg": np.zeros(3)}],
)
def test_chirp_validation(kwargs):
    with pytest.raises(ValueError):
        chirp(10e-3, 20e3, 500, **kwargs)


def test_chirp_180_inverts_band():
    c = chirp(2e-3, 100e3, 2000, smoothing=10, q=5)
    final = simulate([c], np.linspace(-40e3, 40e3, 41)).final
    assert (final[:, 2] < -0.95).all()


def test_presets_verbatim():
    assert "10" not in COMPOSITE_PRESETS
    assert len(COMPOSITE_PRESETS) == 15
    assert COMPOSITE_PRESETS["4"] == ((180, 360, 180, 90), (97.2, 291.5, 97.2, 0))
    assert COMPOSITE_PRESETS["16"] == (
        (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144),
        (180, 0) * 6,
    )
    assert all(len(f) == len(p) for f, p in COMPOSITE_PRESETS.values())


def test_composite_segments():
    segs = composite(10e-6, (90, 180), (0, 120))
    assert [s.n_steps for s in segs] == [90, 180]
    assert segs[0].dt == pytest.approx(10e-6 / 90)
    np.testing.assert_allclose(segs[1].amp, 25e3)
    np.testing.assert_allclose(segs[1].phase, np.deg2rad(120))


def test_composite_180_inverts_on_resonance():
    final = simulate(composite(10e-6, *COMPOSITE_PRESETS["13"]), [0.0]).final[0]
    np.testing.assert_allclose(final, [0, 0, -1], atol=1e-9)


@pytest.mark.parametrize("flips, phases", [((90, 90), (0,)), ((90.5,), (0,)), ((0,), (0,))])
def test_composite_validation(flips, phases):
    with pytest.raises(ValueError):
        composite(10e-6, flips, phases)


SHAPE = """##TITLE= test
##JCAMP-DX= 5.00 Bruker JCAMP library
$$ comment
##NPOINTS= 4
##XYPOINTS= (XY..XY)
 0.0, 0.0
 50.0, 90.0
 100.0, 180.0
 50.0, 270.0
##END=
"""


def test_read_shape(tmp_path):
    p = tmp_path / "pulse.txt"
    p.write_text(SHAPE)
    s = read_bruker_shape(p, 4e-6, 10e3)
    np.testing.assert_allclose(s.amp, [0, 5e3, 10e3, 5e3])
    np.testing.assert_allclose(s.phase, np.deg2rad([0, 90, 180, 270]))
    assert s.dt == pytest.approx(1e-6)
    assert s.name == "pulse"


def test_read_shape_windows_line_endings(tmp_path):  # Review Focus 1
    text = SHAPE.replace(" 50.0, 90.0\n", "\n 50.0, 90.0   \n\n").replace("\n", "\r\n")
    p = tmp_path / "w.txt"
    p.write_bytes(text.encode())
    np.testing.assert_allclose(read_bruker_shape(p, 4e-6, 10e3).amp, [0, 5e3, 10e3, 5e3])


def test_read_shape_malformed_line(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text(SHAPE.replace(" 100.0, 180.0", " abc, 1"))
    with pytest.raises(ValueError, match="line 8"):
        read_bruker_shape(p, 4e-6, 10e3)


def test_read_shape_missing_xypoints(tmp_path):
    p = tmp_path / "none.txt"
    p.write_text("##TITLE= x\n##END=\n")
    with pytest.raises(ValueError, match="XYPOINTS"):
        read_bruker_shape(p, 4e-6, 10e3)


def test_with_peak_rescales_amplitude():
    c = chirp(1e-3, 50e3, 200, smoothing=10)
    scaled = c.with_peak(1234.0)
    assert scaled.amp.max() == pytest.approx(1234.0)
    np.testing.assert_allclose(scaled.amp / scaled.amp.max(), c.amp / c.amp.max())
    np.testing.assert_array_equal(scaled.phase, c.phase)
    assert scaled.dt == c.dt
    assert scaled.name == c.name


def test_with_peak_needs_rf():
    with pytest.raises(ValueError, match="amp"):
        delay(1e-3, 10).with_peak(100.0)


def test_hs_shape():
    # sech amplitude, phase phi0·ln cosh, βT = 10.6.
    T, bw, n = 2e-3, 100e3, 401
    s = hs(T, bw, n, rf_max_hz=5e3, phase_deg=90)
    t = np.linspace(-T / 2, T / 2, n)
    np.testing.assert_allclose(s.amp, 5e3 / np.cosh(10.6 * t / T))
    phi0 = np.pi * bw * T / 10.6
    np.testing.assert_allclose(s.phase, np.pi / 2 + phi0 * np.log(np.cosh(10.6 * t / T)))
    assert s.dt == pytest.approx(T / n)
    assert s.n_steps == n


def test_hs_sweeps_bandwidth():
    T, bw = 2e-3, 100e3
    s = hs(T, bw, 2001, rf_max_hz=5e3)
    freq = np.gradient(s.phase, np.linspace(-T / 2, T / 2, 2001)) / (2 * np.pi)
    assert freq[-1] == pytest.approx(bw / 2 * np.tanh(5.3), rel=1e-3)
    assert freq[0] == pytest.approx(-bw / 2 * np.tanh(5.3), rel=1e-3)
