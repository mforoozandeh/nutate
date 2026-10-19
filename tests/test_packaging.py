import tomllib
from pathlib import Path

PYPROJECT = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())


def test_imageio_ffmpeg_floor_bundles_ffmpeg_on_apple_silicon():
    # 0.5.x has no macOS arm64 wheel: the sdist installs without the bundled ffmpeg binary.
    assert "imageio-ffmpeg>=0.6" in PYPROJECT["project"]["dependencies"]


def test_pyvista_floor_caps_vtk():
    # 0.44 does not cap vtk: a fresh install resolves VTK 9.7, which it cannot import
    # (vtkCapsuleSource is gone). 0.45+ pin a compatible VTK.
    assert "pyvista>=0.45" in PYPROJECT["project"]["dependencies"]


def test_version_matches_pyproject():
    import nutate

    assert nutate.__version__ == PYPROJECT["project"]["version"]


def test_citation_version_matches_pyproject():
    citation = (Path(__file__).parent.parent / "CITATION.cff").read_text()
    assert f"\nversion: {PYPROJECT['project']['version']}\n" in citation


def test_project_urls_for_pypi():
    urls = PYPROJECT["project"]["urls"]
    assert {"Homepage", "Repository", "Issues", "Changelog"} <= urls.keys()


def test_readme_links_are_absolute():  # PyPI renders the README without the repo's files
    import re

    readme = (Path(__file__).parent.parent / "README.md").read_text()
    targets = re.findall(r"\]\(([^)]+)\)", readme)
    assert targets
    assert [t for t in targets if not t.startswith("https://")] == []
