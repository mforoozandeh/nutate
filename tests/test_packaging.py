import tomllib
from pathlib import Path

PYPROJECT = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())


def test_imageio_ffmpeg_floor_bundles_ffmpeg_on_apple_silicon():
    # 0.5.x has no macOS arm64 wheel: the sdist installs without the bundled ffmpeg binary.
    assert "imageio-ffmpeg>=0.6" in PYPROJECT["project"]["dependencies"]
