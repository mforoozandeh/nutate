"""Command-line plumbing shared by the example scripts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nutate as nt

_SNAPSHOT_KEYS = ("offsets", "camera", "mode", "trail", "inset", "inset_phase")


def parse_args(
    description: str,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Common options: --out DIR, --theme, --show, --quick, plus any ``add_arguments`` adds."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    if add_arguments is not None:
        add_arguments(parser)
    parser.add_argument("--out", default="output", help="output directory (default: output)")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--show", action="store_true", help="open the interactive viewer")
    parser.add_argument("--quick", action="store_true", help="tiny, short render (for tests)")
    return parser.parse_args()


def finish(traj: nt.Trajectory, name: str, args: argparse.Namespace, **render_kwargs) -> None:
    """Show ``traj`` interactively, or write ``<out>/<name>.mp4`` and ``<out>/<name>.png``."""
    render_kwargs.setdefault("style", nt.Style(theme=args.theme))
    if args.show:
        show_kwargs = {k: v for k, v in render_kwargs.items() if k in _SNAPSHOT_KEYS}
        nt.show(traj, style=render_kwargs["style"], **show_kwargs)
        return
    if args.quick:
        render_kwargs.update(duration=0.5, hold=0, size=(320, 240))
    out = Path(args.out)
    video = nt.render(traj, out / f"{name}.mp4", **render_kwargs)
    still_kwargs = {k: v for k, v in render_kwargs.items() if k in _SNAPSHOT_KEYS}
    still = nt.snapshot(
        traj,
        out / f"{name}.png",
        size=render_kwargs.get("size", (1920, 1080)),
        style=render_kwargs["style"],
        **still_kwargs,
    )
    print(f"wrote {video} and {still}")


def b1_axes(
    args: argparse.Namespace, bandwidth: float, n_offsets: int = 101
) -> tuple[np.ndarray, np.ndarray]:
    """``n_offsets`` offsets across ±bandwidth/2 and 101 B1 scales 0.5–1.5 (11 each with
    --quick). Sequences with fast phase variation across the band need dense offsets."""
    n_scales = 11 if args.quick else 101
    n_offsets = 11 if args.quick else n_offsets
    return np.linspace(-bandwidth / 2, bandwidth / 2, n_offsets), np.linspace(0.5, 1.5, n_scales)


def finish_b1(
    name: str,
    args: argparse.Namespace,
    sequence: list[nt.Segment],
    offsets: np.ndarray,
    scales: np.ndarray,
    *,
    rf_ref: float,
    sign: float,
    phase_cycle: nt.PhaseCycle | None = None,
) -> None:
    """Write ``<out>/<name>_map.png``, ``_surface.png`` and ``_phase.png`` (or --show them)."""
    bmap = nt.b1_map(sequence, offsets, scales, rf_ref=rf_ref, phase_cycle=phase_cycle)
    dphi = nt.b1_phase_sensitivity(sequence, scales, phase_cycle=phase_cycle)
    figures = {
        "map": nt.plot_b1_map(bmap, "y", sign=sign),
        "surface": nt.plot_b1_map(bmap, "y", sign=sign, surface=True),
        "phase": nt.plot_b1_phase(scales, dphi),
    }
    if args.show:
        plt.show()
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix, fig in figures.items():
        fig.savefig(out / f"{name}_{suffix}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {out / name}_{{map,surface,phase}}.png")
