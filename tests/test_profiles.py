import numpy as np
import pytest

from nutate import b1_map, hard, plot_b1_map, plot_b1_phase, plot_map, plot_profiles, simulate


@pytest.fixture
def traj():
    return simulate([hard(10e-6, n_steps=50)], np.linspace(-5e4, 5e4, 11))


def test_plot_profiles_panels(traj):
    fig = plot_profiles(traj)
    assert [a.get_ylabel() for a in fig.axes] == ["Ix", "Iy", "Iz", "|Ixy|", "phase/π"]
    assert fig.axes[-1].get_xlabel() == "offset (kHz)"
    f = traj.final
    np.testing.assert_allclose(
        fig.axes[4].lines[0].get_ydata(), np.angle(f[:, 1] + 1j * f[:, 0]) / np.pi
    )


def test_plot_map_values(traj):
    fig = plot_map(traj, "y", sign=-1)
    data = np.asarray(fig.axes[0].collections[0].get_array()).reshape(11, 51)
    np.testing.assert_allclose(data, -traj.m[:, :, 1].T)


def test_plot_map_bad_component(traj):
    with pytest.raises(ValueError, match="component"):
        plot_map(traj, "w")


@pytest.fixture
def bmap():
    return b1_map([hard(10e-6)], np.linspace(-5e4, 5e4, 9), np.linspace(0.5, 1.5, 5))


def test_plot_b1_map_contours(bmap):
    fig = plot_b1_map(bmap, "y", sign=-1)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Δ / Ω1(0)"
    assert ax.get_ylabel() == "Ω1 / Ω1(0)"
    np.testing.assert_allclose(ax.get_xlim(), (-2.0, 2.0))  # ±50 kHz / 25 kHz
    np.testing.assert_allclose(ax.get_ylim(), (0.5, 1.5))
    assert fig.axes[1].get_ylabel() == "−My"  # colorbar


def test_plot_b1_map_surface(bmap):
    fig = plot_b1_map(bmap, "z", surface=True)
    ax = fig.axes[0]
    assert ax.name == "3d"
    assert ax.get_zlabel() == "Mz"


def test_plot_b1_map_bad_component(bmap):
    with pytest.raises(ValueError, match="component"):
        plot_b1_map(bmap, "w")


def test_plot_b1_phase():
    scales = np.linspace(0.5, 1.5, 11)
    dphi = np.linspace(-3, 3, 11)
    fig = plot_b1_phase(scales, dphi, delta=0.005)
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), scales)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), dphi)
    assert ax.get_xlabel() == "Ω1 / Ω1(0)"
    assert ax.get_ylabel() == "ΔΦ for ±0.5 % B1 (°)"


def test_plot_b1_map_surface_draws_every_sample():
    # matplotlib's plot_surface defaults to at most 50 × 50 faces; every sample must show.
    dense = b1_map([hard(10e-6, n_steps=1)], np.linspace(-5e4, 5e4, 101), np.linspace(0.5, 1.5, 5))
    fig = plot_b1_map(dense, surface=True)
    fig.canvas.draw()
    assert len(fig.axes[0].collections[0].get_paths()) == (5 - 1) * (101 - 1)
