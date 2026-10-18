"""Command-line plumbing shared by the example scripts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import numpy as np

import nutate as nt


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
    parser.add_argument("--show", action="store_true", help="open interactive windows instead")
    parser.add_argument("--quick", action="store_true", help="tiny, short render (for tests)")
    return parser.parse_args()


def run_options(
    args: argparse.Namespace, *, b1_offsets: int = 101, **video_options: Any
) -> dict[str, Any]:
    """``nt.run`` arguments for --out, --theme, --show and --quick.

    --quick makes tiny, short videos and an 11 × 11 B1 grid. Sequences with fast phase
    variation across the band need dense ``b1_offsets``.
    """
    video_options = {"style": nt.Style(theme=args.theme), **video_options}
    options = {
        "out": args.out,
        "show": args.show,
        "video_options": video_options,
        "b1_offsets": b1_offsets,
    }
    if args.quick:
        video_options.update(duration=0.5, hold=0, size=(320, 240))
        options.update(b1_offsets=11, b1_scales=np.linspace(0.5, 1.5, 11))
    return options


def report(result: nt.RunResult) -> None:
    for path in result.files:
        print(f"wrote {path}")
