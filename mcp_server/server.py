"""Model Context Protocol server for PerfectPixelTool.

Run from the repository root with: ``python -m mcp_server.server``.
The tools use local file paths and return JSON-serialisable metadata.
"""
from __future__ import annotations

import sys
import os
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from mcp.server.fastmcp import FastMCP

# Allow running directly from a checkout without an editable install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from perfect_pixel import get_perfect_pixel, remove_fake_checkerboard as _remove_fake_checkerboard
from perfect_pixel.background_remover import remove_background
from perfect_pixel.app_core.image_io import load_rgb, save_png
from perfect_pixel.app_core.image_io import load_rgba

mcp = FastMCP("PerfectPixelTool")
OUTPUT_DIR = Path("mcp_outputs")
_desktop_process: subprocess.Popen[bytes] | None = None


def _count_components(mask: np.ndarray) -> int:
    """Count 4-connected true regions without requiring OpenCV."""
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        count += 1
        stack = [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return count


def _load(path: str) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    return load_rgb(p)


def _save(arr: np.ndarray, stem: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import uuid
    out = OUTPUT_DIR / f"{stem}_{uuid.uuid4().hex[:10]}.png"
    save_png(out, np.asarray(arr, dtype=np.uint8))
    return out.resolve()


@mcp.tool()
def refine_pixel_art(
    input_path: str,
    sample_method: str = "center",
    refine_intensity: float = 0.25,
    fix_square: bool = True,
) -> dict[str, Any]:
    """Detect a pixel grid and export a perfectly aligned RGB PNG."""
    if sample_method not in {"center", "median", "majority"}:
        raise ValueError("sample_method must be center, median, or majority")
    image = _load(input_path)
    # The algorithm prints diagnostic messages; stdout belongs to MCP JSON-RPC.
    with redirect_stdout(sys.stderr):
        width, height, result = get_perfect_pixel(
            image, sample_method=sample_method,
            refine_intensity=float(refine_intensity), fix_square=fix_square,
        )
    if width is None or height is None or result is None:
        raise ValueError("Could not detect a pixel grid in this image")
    output = _save(result, "refined")
    return {"output_path": str(output), "width": width, "height": height}


@mcp.tool()
def resize_image(input_path: str, width: int, height: int) -> dict[str, Any]:
    """Resize an image with nearest-neighbour sampling, preserving hard pixels."""
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    image = _load(input_path)
    result = np.array(Image.fromarray(image).resize((width, height), Image.Resampling.NEAREST))
    output = _save(result, "resized")
    return {"output_path": str(output), "width": width, "height": height}


@mcp.tool()
def remove_image_background(
    input_path: str,
    mode: str = "color",
    background_color: list[int] | None = None,
) -> dict[str, Any]:
    """Remove a background using the existing color/channel/AI implementation."""
    image = _load(input_path)
    kwargs: dict[str, Any] = {"mode": mode}
    if background_color is not None:
        if len(background_color) != 3:
            raise ValueError("background_color must contain three RGB values")
        kwargs["target_color"] = tuple(int(x) for x in background_color)
    result = remove_background(image, **kwargs)
    output = _save(result, "background_removed")
    return {"output_path": str(output), "width": int(result.shape[1]), "height": int(result.shape[0])}


@mcp.tool()
def remove_fake_checkerboard(
    input_path: str,
    tile_size: int | None = None,
    tolerance: float = 18.0,
    anti_alias: bool = True,
    binary_alpha: bool = False,
    cleanup_islands: bool = True,
    min_component_size: int = 4,
    edge_shrink: int = 1,
) -> dict[str, Any]:
    """Remove a baked white/gray checkerboard and export an RGBA PNG."""
    if tile_size is not None and tile_size < 1:
        raise ValueError("tile_size must be positive or null")
    if tolerance < 0 or tolerance > 100:
        raise ValueError("tolerance must be between 0 and 100")
    if min_component_size < 1:
        raise ValueError("min_component_size must be at least 1")
    if edge_shrink < 0 or edge_shrink > 8:
        raise ValueError("edge_shrink must be between 0 and 8")
    p = Path(input_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    image = load_rgba(p)
    baseline = _remove_fake_checkerboard(
        image, tile_size=tile_size, tolerance=float(tolerance),
        anti_alias=bool(anti_alias), binary_alpha=bool(binary_alpha),
        cleanup_islands=False, min_component_size=1, edge_shrink=int(edge_shrink),
    )
    result = _remove_fake_checkerboard(
        image, tile_size=tile_size, tolerance=float(tolerance),
        anti_alias=bool(anti_alias), binary_alpha=bool(binary_alpha),
        cleanup_islands=bool(cleanup_islands),
        min_component_size=int(min_component_size), edge_shrink=int(edge_shrink),
    )
    output = _save(result, "checkerboard")
    alpha = result[..., 3]
    baseline_alpha = baseline[..., 3]
    return {
        "output_path": str(output), "width": int(result.shape[1]),
        "height": int(result.shape[0]), "mode": "RGBA",
        "transparent_pixels": int(np.count_nonzero(alpha == 0)),
        "partial_alpha_pixels": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
        "removed_island_pixels": int(max(0, np.count_nonzero(baseline_alpha > 0) - np.count_nonzero(alpha > 0))),
        "edge_shrink": int(edge_shrink),
        "remaining_components": int(_count_components(alpha > 0)),
    }


@mcp.tool()
def launch_desktop_app() -> dict[str, Any]:
    """Launch the Perfect Pixel Tool desktop window and return its process id.

    The MCP server remains responsive because the GUI is started as a detached
    child process. Calling this tool again while that child is alive is a no-op.
    """
    global _desktop_process
    if _desktop_process is not None and _desktop_process.poll() is None:
        return {"ok": True, "already_running": True, "pid": _desktop_process.pid}
    python_exe = _ROOT / ".venv" / "Scripts" / "pythonw.exe"
    if not python_exe.is_file():
        python_exe = _ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_exe.is_file():
        python_exe = Path(sys.executable).resolve()
    app_script = _ROOT / "desktop_app.py"
    if not app_script.is_file():
        raise FileNotFoundError(f"Desktop entry point not found: {app_script}")
    env = os.environ.copy()
    env.pop("QT_QPA_PLATFORM", None)
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), str(_ROOT / "src"), env.get("PYTHONPATH", "")])
    # A detached process with no console can suppress Qt's top-level window on
    # some Windows builds. A new process group is sufficient; stdio is already
    # redirected below so the MCP server remains independent.
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    _desktop_process = subprocess.Popen(
        [str(python_exe), str(app_script)], cwd=_ROOT, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, creationflags=creationflags,
    )
    return {"ok": True, "already_running": False, "pid": _desktop_process.pid,
            "executable": str(python_exe), "entry_point": str(app_script)}


@mcp.tool()
def check_desktop_startup(timeout_seconds: int = 120) -> dict[str, Any]:
    """Start the PySide6 desktop app headlessly and verify MainWindow creation.

    This is a diagnostic tool: it does not show a window and exits immediately
    after constructing the main window, so it is safe for MCP automation.
    """
    timeout = max(3, min(int(timeout_seconds), 180))
    code = (
        "import os; from PySide6.QtWidgets import QApplication; "
        "from desktop_app import MainWindow; "
        "app=QApplication([]); w=MainWindow(); "
        "print('MAINWINDOW_OK tabs=' + str(w.tabs.count()), flush=True); os._exit(0)"
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    root = str(_ROOT)
    env["PYTHONPATH"] = os.pathsep.join([root, str(_ROOT / "src"), env.get("PYTHONPATH", "")])
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=root, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": "timeout", "timeout_seconds": timeout,
                "stderr": str(exc)}
    return {
        "ok": proc.returncode == 0 and "MAINWINDOW_OK" in proc.stdout,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-8000:],
    }


@mcp.tool()
def check_splitter_drag() -> dict[str, Any]:
    """Use QtTest to verify both workspace splitter handles resize panels."""
    code = '''
import json, os
from PySide6.QtCore import QPointF, Qt, QCoreApplication, QEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from desktop_app import MainWindow
app = QApplication([])
window = MainWindow(); window.show(); app.processEvents()
splitter = window.workspace_splitter
checks = []
for handle_index in (1, 2):
    handle = splitter.handle(handle_index)
    before = splitter.sizes()
    point = handle.rect().center()
    def send(kind, x):
        event = QMouseEvent(kind, QPointF(x, point.y()), QPointF(x, point.y()),
                            Qt.LeftButton, Qt.LeftButton if kind != QEvent.Type.MouseButtonRelease else Qt.NoButton,
                            Qt.NoModifier)
        QCoreApplication.sendEvent(handle, event)
    send(QEvent.Type.MouseButtonPress, point.x())
    send(QEvent.Type.MouseMove, point.x() + 80)
    send(QEvent.Type.MouseButtonRelease, point.x() + 80)
    app.processEvents()
    after = splitter.sizes()
    checks.append({"handle": handle_index, "before": before, "after": after,
                   "changed": before != after})
print(json.dumps({"checks": checks}), flush=True)
os._exit(0)
'''
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), str(_ROOT / "src"), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=60)
    try:
        result = __import__("json").loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-4000:]}
    result["ok"] = proc.returncode == 0 and all(item["changed"] for item in result["checks"])
    return result


@mcp.tool()
def inspect_image(input_path: str) -> dict[str, Any]:
    """Return image metadata useful for debugging transparency and sprite sheets."""
    p = Path(input_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    with Image.open(p) as image:
        image.load()
        result: dict[str, Any] = {
            "path": str(p), "format": image.format, "mode": image.mode,
            "width": image.width, "height": image.height,
        }
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
            result.update({
                "alpha_min": int(alpha.min()), "alpha_max": int(alpha.max()),
                "transparent_pixels": int(np.count_nonzero(alpha == 0)),
                "opaque_pixels": int(np.count_nonzero(alpha == 255)),
                "partial_alpha_pixels": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
            })
        return result


@mcp.tool()
def check_project_health(include_desktop: bool = False) -> dict[str, Any]:
    """Check Python syntax and imports before parallel feature work begins."""
    files = sorted((_ROOT / "src").rglob("*.py")) + sorted(_ROOT.glob("*.py"))
    syntax_errors: list[str] = []
    for file in files:
        try:
            compile(file.read_text(encoding="utf-8"), str(file), "exec")
        except (OSError, SyntaxError) as exc:
            syntax_errors.append(f"{file}: {exc}")
    imports: dict[str, str] = {}
    modules = ["perfect_pixel", "perfect_pixel.background_remover"]
    if include_desktop:
        modules.append("desktop_app")
    code = "import importlib, json; mods=%r; out={}; " \
        "\nfor m in mods:\n  " \
        "\n  try: importlib.import_module(m); out[m]='ok'" \
        "\n  except Exception as e: out[m]=f'{type(e).__name__}: {e}'" \
        "\nprint(json.dumps(out))" % modules
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), str(_ROOT / "src"), env.get("PYTHONPATH", "")])
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        if proc.stdout.strip():
            import json
            imports = json.loads(proc.stdout.strip().splitlines()[-1])
        elif proc.stderr:
            imports["__process__"] = proc.stderr[-1000:]
    except Exception as exc:
        imports["__process__"] = f"{type(exc).__name__}: {exc}"
    return {
        "ok": not syntax_errors and all(value == "ok" for value in imports.values()),
        "python": sys.executable, "files_checked": len(files),
        "syntax_errors": syntax_errors, "imports": imports,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
