"""Exercise the real stdio protocol and inspect generated PNG files.

Run: .venv/Scripts/python.exe -m mcp_server.smoke_test
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="perfectpixel-mcp-") as directory:
        workdir = Path(directory)
        source = workdir / "input.png"
        image = Image.new("RGB", (64, 64), "white")
        image.paste((180, 50, 50), (16, 16, 48, 48))
        image.save(source)
        checker_source = workdir / "checker.png"
        checker = Image.new("RGBA", (40, 32), (238, 238, 238, 255))
        pixels = checker.load()
        for y in range(32):
            for x in range(40):
                if (x // 4 + y // 4) % 2:
                    pixels[x, y] = (205, 205, 205, 255)
        for y in range(10, 23):
            for x in range(12, 28):
                pixels[x, y] = (220, 40, 30, 255)
        checker.save(checker_source)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(ROOT / "mcp_server" / "server.py"),
            cwd=workdir, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stderr_task = asyncio.create_task(process.stderr.read())
        next_id = 0

        async def request(method, params=None, *, notification=False):
            nonlocal next_id
            next_id += 1
            payload = {"jsonrpc": "2.0", "method": method}
            if not notification:
                payload["id"] = next_id
            if params is not None:
                payload["params"] = params
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()
            if notification:
                return
            while True:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=180)
                if not line:
                    raise RuntimeError("Server exited before replying")
                try:
                    reply = json.loads(line)
                except ValueError as exc:
                    raise AssertionError(f"Non-JSON data on MCP stdout: {line!r}") from exc
                if reply.get("id") == next_id:
                    assert "error" not in reply, reply
                    return reply["result"]

        async def call(name, arguments):
            result = await request("tools/call", {"name": name, "arguments": arguments})
            assert not result.get("isError"), result
            metadata = result.get("structuredContent")
            if metadata is None:
                metadata = json.loads(result["content"][0]["text"])
            if name == "check_desktop_startup":
                assert metadata.get("ok") is True, metadata
                print(f"PASS {name}: desktop startup")
                return metadata
            if name == "inspect_image":
                print(f"PASS {name}: {metadata['mode']} {metadata['width']} x {metadata['height']}")
                return metadata
            if name == "check_project_health":
                return metadata
            with Image.open(metadata["output_path"]) as output:
                assert output.size == (metadata["width"], metadata["height"]), metadata
                output.load()
            print(f"PASS {name}: {metadata['width']} x {metadata['height']}")
            return metadata

        try:
            result = await request("initialize", {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "perfectpixel-smoke", "version": "1.0"},
            })
            print(f"PASS initialize: {result['serverInfo']['name']}")
            await request("notifications/initialized", notification=True)
            result = await request("tools/list")
            names = {tool["name"] for tool in result["tools"]}
            assert names == {"resize_image", "refine_pixel_art", "remove_image_background", "remove_fake_checkerboard", "cleanup_background_residuals", "launch_desktop_app", "check_desktop_startup", "check_splitter_drag", "inspect_image", "check_project_health"}, names
            print("PASS tools/list: " + ", ".join(sorted(names)))
            await call("resize_image", {"input_path": str(source), "width": 32, "height": 48})
            metadata = await call("inspect_image", {"input_path": str(source)})
            assert metadata["mode"] == "RGB" and metadata["width"] == 64
            background = await call("remove_image_background", {
                "input_path": str(source), "background_color": [255, 255, 255],
            })
            with Image.open(background["output_path"]) as output:
                assert output.mode == "RGBA"
                assert output.getpixel((0, 0))[3] == 0
                assert output.getpixel((32, 32))[3] == 255
            checker_result = await call("remove_fake_checkerboard", {"input_path": str(checker_source), "tile_size": 4, "binary_alpha": True})
            assert checker_result["transparent_pixels"] > 0
            await call("refine_pixel_art", {"input_path": str(ROOT / "images" / "avatar.png")})
            invalid = await request("tools/call", {"name": "resize_image", "arguments": {
                "input_path": str(source), "width": 0, "height": 10,
            }})
            assert invalid.get("isError"), invalid
            await request("ping")
            print("PASS invalid arguments return an error; server remains responsive")
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.terminate()
                await process.wait()
            stderr = (await stderr_task).decode(errors="replace")
            if sys.exc_info()[0] is not None:
                print(stderr, file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
