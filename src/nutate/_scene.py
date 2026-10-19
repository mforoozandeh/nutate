"""The Bloch-sphere scene: built once, then moved to any simulation step."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pyvista as pv

from nutate._format import si_scale
from nutate.simulate import Trajectory

# Starting values; final look agreed at the milestone-2 review.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "background": "#ffffff",
        "text": "#222222",
        "sphere": "#c8d3e0",
        "grid": "#8a94a3",
        "cmap": "berlin",  # dark centre reads on white
        "series": "#2a78d6",  # inset amplitude line
    },
    "dark": {
        "background": "#0f1115",
        "text": "#e6e6e6",
        "sphere": "#9fb3c8",
        "grid": "#9aa3b2",
        "cmap": "coolwarm",  # light centre reads on dark
        "series": "#5b9cf0",
    },
}

_FIT_RADIUS = 1.45  # sphere + axis labels
_VIEW_ANGLE = 25.0


@dataclass(frozen=True)
class Style:
    """Fine-tuning of the scene's look. ``cmap=None`` takes the theme's default; ``color``
    paints every arrow one colour (no colorbar)."""

    cmap: str | None = None
    color: str | None = None
    sphere_opacity: float = 0.12
    font: str = "times"
    arrow_shaft_radius: float = 0.015
    arrow_tip_radius: float = 0.05
    arrow_tip_length: float = 0.15
    line_width: float = 3.0


@dataclass(frozen=True)
class Camera:
    """Viewpoint in degrees: azimuth about +z measured from −y toward +x, elevation above
    the xy-plane.

    ``orbit`` adds that many degrees of azimuth across a video.
    """

    azimuth: float = 45.0
    elevation: float = 22.5
    orbit: float = 0.0
    orthographic: bool = False
    zoom: float = 1.0


def _camera_position(
    camera: Camera, distance: float, fraction: float = 0.0
) -> tuple[float, float, float]:
    az = math.radians(camera.azimuth + camera.orbit * fraction)
    el = math.radians(camera.elevation)
    return (
        distance * math.sin(az) * math.cos(el),
        -distance * math.cos(az) * math.cos(el),
        distance * math.sin(el),
    )


def _trail_window(step: int, n_steps: int, trail: float, max_vertices: int = 200) -> np.ndarray:
    """Steps drawn in the trail ending at ``step``: the last ``trail`` fraction of the
    sequence, evenly subsampled to at most ``max_vertices``. Empty when ``trail`` is 0."""
    if trail <= 0:
        return np.array([], dtype=int)
    start = max(0, step - round(trail * n_steps))
    if step - start + 1 <= max_vertices:
        return np.arange(start, step + 1)
    return np.round(np.linspace(start, step, max_vertices)).astype(int)


def _polylines(points: np.ndarray) -> pv.PolyData:
    """One polyline per row of ``points`` (shape ``(n_lines, n_points, 3)``)."""
    n_lines, n_points, _ = points.shape
    mesh = pv.PolyData(points.reshape(-1, 3))
    ids = np.arange(n_lines * n_points).reshape(n_lines, n_points)
    mesh.lines = np.column_stack([np.full(n_lines, n_points), ids]).ravel()
    return mesh


def _circle(points: np.ndarray) -> pv.PolyData:
    return pv.lines_from_points(points, close=True)


def _latitude(lat_deg: float) -> pv.PolyData:
    s = np.linspace(0, 2 * np.pi, 181)[:-1]
    lat = np.radians(lat_deg)
    return _circle(
        np.column_stack(
            [np.cos(lat) * np.cos(s), np.cos(lat) * np.sin(s), np.full_like(s, np.sin(lat))]
        )
    )


def _meridian(az_deg: float) -> pv.PolyData:
    s = np.linspace(0, 2 * np.pi, 181)[:-1]
    az = np.radians(az_deg)
    return _circle(np.column_stack([np.cos(az) * np.cos(s), np.sin(az) * np.cos(s), np.sin(s)]))


class _Scene:
    """All actors for one trajectory; :meth:`update` moves them to a step."""

    def __init__(
        self,
        plotter: pv.Plotter,
        traj: Trajectory,
        *,
        offsets: np.ndarray,
        style: Style,
        camera: Camera,
        theme: Literal["light", "dark"] = "light",
        mode: Literal["arrows", "ribbon", "both"] = "arrows",
        trail: float = 0.1,
        inset: bool = True,
        inset_phase: bool = False,
    ) -> None:
        if theme not in THEMES:
            raise ValueError(f"theme must be one of {sorted(THEMES)}, got {theme!r}")
        if mode not in ("arrows", "ribbon", "both"):
            raise ValueError(f"mode must be 'arrows', 'ribbon' or 'both', got {mode!r}")
        if not 0 <= trail <= 1:
            raise ValueError(f"trail must be a fraction in [0, 1], got {trail}")
        if mode != "arrows" and len(offsets) < 2:
            raise ValueError("a ribbon needs at least 2 displayed offsets")
        self.plotter = plotter
        self.traj = traj
        self.idx = np.asarray(offsets)
        self.style = style
        self.camera = camera
        self.mode = mode
        self.trail = trail
        self.theme = THEMES[theme]
        height = plotter.window_size[1]
        self._font_size = max(8, round(24 * height / 1080))

        factor, self._offset_unit = si_scale(np.abs(traj.offsets).max(), "Hz")
        self._offset_values = traj.offsets[self.idx] * factor
        limit = np.abs(traj.offsets).max() * factor
        self._clim = (-limit, limit) if limit > 0 else (-1.0, 1.0)
        self._cmap = style.cmap or self.theme["cmap"]
        if style.color is not None:
            rgb = np.tile(mcolors.to_rgb(style.color), (self.idx.size, 1))
        else:
            norm = mcolors.Normalize(*self._clim)
            rgb = mpl.colormaps[self._cmap](norm(self._offset_values))[:, :3]
        self._rgb = 255 * rgb
        self._background = 255 * np.array(mcolors.to_rgb(self.theme["background"]))

        plotter.set_background(self.theme["background"])
        plotter.enable_depth_peeling()
        plotter.enable_anti_aliasing("ssaa")
        self._add_sphere()
        self._add_axes()
        coloured = []
        if trail > 0:
            self._add_trails()
        if mode != "arrows":
            coloured.append(self._add_ribbon())
        if mode != "ribbon":
            coloured.append(self._add_arrows())
        if style.color is None:
            self._add_scalar_bar(coloured[-1])
        self._cursors: list[tuple[pv.charts.LinePlot2D, float]] = []
        self._t_factor, self._t_unit = si_scale(traj.t[-1], "s")
        # Shift the sphere left to leave the bottom-right free for the inset.
        self._window_center = (0.2, 0.0) if inset else (0.0, 0.0)
        if inset:
            self._add_inset(inset_phase)
        self._text = plotter.add_text(
            "",
            position=(0.03, 0.85),
            viewport=True,
            font_size=self._font_size,
            color=self.theme["text"],
            font=style.font,
        )

        self._distance = _FIT_RADIUS / math.sin(math.radians(_VIEW_ANGLE / 2)) / camera.zoom
        if camera.orthographic:
            plotter.enable_parallel_projection()
        self._place_camera(0.0)
        self.update(0)

    # -- static parts ---------------------------------------------------------------
    def _add_sphere(self) -> None:
        p, th, st = self.plotter, self.theme, self.style
        p.add_mesh(
            pv.Sphere(
                radius=0.99, theta_resolution=96, phi_resolution=96
            ),  # r=1 would z-fight trails
            color=th["sphere"],
            opacity=st.sphere_opacity,
            smooth_shading=True,
            specular=0.4,
            specular_power=20,
        )
        faint = pv.merge(
            [_latitude(lat) for lat in (-60, -30, 30, 60)]
            + [_meridian(az) for az in (30, 60, 90, 120, 150)]
        )
        p.add_mesh(faint, color=th["grid"], line_width=1, opacity=0.25)
        strong = pv.merge([_latitude(0), _meridian(0)])
        p.add_mesh(strong, color=th["grid"], line_width=2, opacity=0.6)

    def _add_axes(self) -> None:
        p, th = self.plotter, self.theme
        for axis in np.eye(3):
            p.add_mesh(
                pv.Line(-1.2 * axis, 1.2 * axis), color=th["text"], line_width=1.5, opacity=0.6
            )
        points = np.vstack([1.3 * np.eye(3), -1.3 * np.eye(3)])
        labels = ["x", "y", "z", "−x", "−y", "−z"]
        p.add_point_labels(
            points,
            labels,
            italic=True,
            bold=False,
            font_size=self._font_size,
            text_color=th["text"],
            font_family=self.style.font,
            shape=None,
            show_points=False,
            always_visible=True,
        )

    # -- arrows -----------------------------------------------------------------------
    def _arrow_glyphs(self, step: int) -> pv.PolyData:
        src = pv.PolyData(np.zeros((self.idx.size, 3)))
        src["vectors"] = self.traj.m[step, self.idx]
        src["offset"] = self._offset_values
        return src.glyph(orient="vectors", scale="vectors", geom=self._arrow_geom)

    def _add_coloured(self, mesh: pv.PolyData, **kwargs) -> pv.Actor:
        """Add ``mesh`` coloured by its ``offset`` scalars, or in the single style colour."""
        if self.style.color is not None:
            return self.plotter.add_mesh(mesh, color=self.style.color, **kwargs)
        return self.plotter.add_mesh(
            mesh,
            scalars="offset",
            cmap=self._cmap,
            clim=self._clim,
            show_scalar_bar=False,
            **kwargs,
        )

    def _add_arrows(self) -> pv.Actor:
        st = self.style
        self._arrow_geom = pv.Arrow(
            tip_length=st.arrow_tip_length,
            tip_radius=st.arrow_tip_radius,
            shaft_radius=st.arrow_shaft_radius,
        )
        self._arrows = self._arrow_glyphs(0)
        return self._add_coloured(self._arrows, smooth_shading=True)

    # -- trails and ribbon -------------------------------------------------------------
    def _trail_mesh(self, step: int) -> pv.PolyData:
        window = _trail_window(step, self.traj.t.size - 1, self.trail)
        if window.size < 2:  # nothing to draw yet: a zero-length line
            window = np.array([step, step])
        points = self.traj.m[window][:, self.idx].transpose(1, 0, 2)
        mesh = _polylines(points)
        # Fade toward the background rather than with opacity: overlapping translucent
        # tube segments leave dark beads at every vertex.
        fade = np.linspace(0.0, 1.0, window.size)[None, :, None]
        rgb = self._background + (self._rgb[:, None, :] - self._background) * fade
        mesh["rgb"] = np.round(rgb).astype(np.uint8).reshape(-1, 3)
        return mesh

    def _add_trails(self) -> None:
        self._trails = self._trail_mesh(0)
        self.plotter.add_mesh(
            self._trails,
            scalars="rgb",
            rgb=True,
            line_width=self.style.line_width,  # flat lines: tube joints bead at vertices
        )

    def _ribbon_mesh(self, step: int) -> pv.PolyData:
        order = np.argsort(self._offset_values)
        mesh = _polylines(self.traj.m[step, self.idx[order]][None])
        mesh["offset"] = self._offset_values[order]
        return mesh

    def _add_ribbon(self) -> pv.Actor:
        self._ribbon = self._ribbon_mesh(0)
        return self._add_coloured(
            self._ribbon, render_lines_as_tubes=True, line_width=1.5 * self.style.line_width
        )

    def _add_scalar_bar(self, actor: pv.Actor) -> None:
        st = self.style
        self.plotter.add_scalar_bar(
            title=f"offset ({self._offset_unit})",
            mapper=actor.mapper,
            vertical=True,
            position_x=0.9,
            position_y=0.5,
            width=0.05,
            height=0.4,
            color=self.theme["text"],
            font_family=st.font,
            title_font_size=self._font_size,
            label_font_size=max(8, round(0.8 * self._font_size)),
            n_labels=3,
            fmt="%.3g",
        )

    # -- per step ---------------------------------------------------------------------
    # -- pulse inset --------------------------------------------------------------------
    def _chart(self, loc: tuple[float, float], x_label: str, y_label: str) -> pv.Chart2D:
        chart = pv.Chart2D(size=(0.35, 0.18), loc=loc, x_label=x_label, y_label=y_label, grid=False)
        chart.background_color = (0, 0, 0, 0)
        chart.border_color = (0, 0, 0, 0)
        text_rgb = mcolors.to_rgb(self.theme["text"])
        size = max(6, round(0.6 * self._font_size))
        for axis in (chart.x_axis, chart.y_axis):
            axis.pen.color = self.theme["text"]
            axis.label_size = size
            axis.tick_label_size = size
            axis.GetTitleProperties().SetColor(text_rgb)
            axis.GetLabelProperties().SetColor(text_rgb)
        self._overlay.add_chart(chart)
        return chart

    def _add_inset(self, with_phase: bool) -> None:
        """Drive amplitude (and optionally wrapped phase) of the whole sequence, with a cursor."""
        # Charts go on an overlay layer: under the main renderer's SSAA pass they would be
        # laid out in the supersampled buffer and come out shrunk and misplaced.
        self._overlay = pv.Renderer(self.plotter)
        self._overlay.SetLayer(1)
        self._overlay.SetInteractive(False)
        self.plotter.ren_win.SetNumberOfLayers(2)
        self.plotter.ren_win.AddRenderer(self._overlay)
        seq = self.traj.sequence
        t = self.traj.t[1:] * self._t_factor
        amp = np.concatenate([s.drive for s in seq])
        a_factor, a_unit = si_scale(amp.max(), "Hz")
        t_label = f"t ({self._t_unit})"
        panels = [(f"B1 ({a_unit})", amp * a_factor)]
        if with_phase:
            phase = np.rad2deg(np.mod(np.concatenate([s.phase for s in seq]), 2 * np.pi))
            panels.append(("phase (°)", phase))
        locs = [(0.6, 0.25), (0.6, 0.04)] if with_phase else [(0.6, 0.04)]
        for (y_label, y), loc in zip(panels, locs):
            last = loc == locs[-1]
            chart = self._chart(loc, t_label if last else "", y_label)
            top = 360.0 if y_label.startswith("phase") else max(1.1 * y.max(), 1e-12)
            if y_label.startswith("phase"):
                chart.scatter(t, y, color=self.theme["series"], size=2)
            else:
                chart.line(t, y, color=self.theme["series"], width=1.5)
            chart.x_range = [0.0, t[-1]]
            chart.y_range = [0.0, top]
            cursor = chart.line([0.0, 0.0], [0.0, top], color=self.theme["text"], width=1.5)
            self._cursors.append((cursor, top))

    def _segment_name(self, step: int) -> str:
        for name, start, stop in self.traj.segments:
            if start < step <= stop:
                return name
        return self.traj.segments[0][0]

    def update(self, step: int, fraction: float | None = None) -> None:
        """Move every dynamic actor to simulation ``step``.

        ``fraction`` ∈ [0, 1] of a video also places the camera (orbit); ``None`` leaves the
        camera alone, so an interactive user's view survives scrubbing.
        """
        if self.mode != "ribbon":
            self._arrows.copy_from(self._arrow_glyphs(step))
        if self.mode != "arrows":
            self._ribbon.copy_from(self._ribbon_mesh(step))
        if self.trail > 0:
            self._trails.copy_from(self._trail_mesh(step))
        for cursor, top in self._cursors:
            tc = self.traj.t[step] * self._t_factor
            cursor.update([tc, tc], [0.0, top])
        t = self.traj.t[step] * self._t_factor
        self._text.SetInput(f"t = {t:.2f} {self._t_unit}\n{self._segment_name(step)}")
        if fraction is not None:
            self._place_camera(fraction)

    def _place_camera(self, fraction: float) -> None:
        cam = self.plotter.camera
        self.plotter.camera_position = [
            _camera_position(self.camera, self._distance, fraction),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        cam.view_angle = _VIEW_ANGLE
        cam.SetWindowCenter(*self._window_center)
        if self.camera.orthographic:
            cam.parallel_scale = _FIT_RADIUS / self.camera.zoom
        self.plotter.renderer.ResetCameraClippingRange()

    def screenshot(self) -> np.ndarray:
        self.plotter.render()
        return self.plotter.screenshot(return_img=True)
