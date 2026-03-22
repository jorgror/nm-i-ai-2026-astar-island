"""Simple SVG visualizations for maps and probability tensors."""

from __future__ import annotations

import math
from pathlib import Path

from .constants import CLASS_COLORS, GRID_COLORS
from .models import Grid2D, Tensor3D
from .scoring import cell_entropy


def write_initial_grid_svg(grid: Grid2D, path: str | Path, cell_size: int = 12) -> None:
    write_categorical_grid_svg(grid, GRID_COLORS, path=path, cell_size=cell_size)


def write_argmax_svg(tensor: Tensor3D, path: str | Path, cell_size: int = 12) -> None:
    argmax_grid = tensor_argmax_grid(tensor)
    write_categorical_grid_svg(argmax_grid, CLASS_COLORS, path=path, cell_size=cell_size)


def write_entropy_svg(tensor: Tensor3D, path: str | Path, cell_size: int = 12) -> None:
    entropy_grid = tensor_entropy_grid(tensor)
    write_heatmap_svg(entropy_grid, path=path, cell_size=cell_size)


def tensor_argmax_grid(tensor: Tensor3D) -> Grid2D:
    argmax: Grid2D = []
    for row in tensor:
        argmax_row: list[int] = []
        for probs in row:
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            argmax_row.append(int(best_idx))
        argmax.append(argmax_row)
    return argmax


def tensor_entropy_grid(tensor: Tensor3D) -> list[list[float]]:
    return [[cell_entropy(cell) for cell in row] for row in tensor]


def write_categorical_grid_svg(
    grid: Grid2D,
    palette: dict[int, str],
    *,
    path: str | Path,
    cell_size: int = 12,
) -> None:
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    svg_width = width * cell_size
    svg_height = height * cell_size

    parts: list[str] = [_svg_open(svg_width, svg_height)]

    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            color = palette.get(value, "#000000")
            parts.append(
                f'<rect x="{x * cell_size}" y="{y * cell_size}" '
                f'width="{cell_size}" height="{cell_size}" fill="{color}" />'
            )

    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def write_heatmap_svg(
    values: list[list[float]],
    *,
    path: str | Path,
    cell_size: int = 12,
) -> None:
    height = len(values)
    width = len(values[0]) if height > 0 else 0

    flat = [v for row in values for v in row]
    v_min = min(flat) if flat else 0.0
    v_max = max(flat) if flat else 1.0
    if math.isclose(v_min, v_max):
        v_max = v_min + 1.0

    svg_width = width * cell_size
    svg_height = height * cell_size

    parts: list[str] = [_svg_open(svg_width, svg_height)]

    for y, row in enumerate(values):
        for x, value in enumerate(row):
            norm = (value - v_min) / (v_max - v_min)
            color = _heat_color(norm)
            parts.append(
                f'<rect x="{x * cell_size}" y="{y * cell_size}" '
                f'width="{cell_size}" height="{cell_size}" fill="{color}" />'
            )

    parts.append("</svg>")
    _write_text(path, "\n".join(parts))


def _svg_open(width: int, height: int) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )


def _heat_color(norm: float) -> str:
    norm = max(0.0, min(1.0, norm))
    # Blue -> yellow -> red for easy contrast.
    if norm < 0.5:
        t = norm * 2.0
        r = int(45 + (255 - 45) * t)
        g = int(106 + (220 - 106) * t)
        b = int(193 + (80 - 193) * t)
    else:
        t = (norm - 0.5) * 2.0
        r = int(255)
        g = int(220 - (220 - 70) * t)
        b = int(80 - (80 - 40) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _write_text(path: str | Path, content: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
