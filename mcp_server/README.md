# PerfectPixelTool MCP

Use Python 3.10 or newer. To create a project environment and install only the
dependencies needed by these tools (without the desktop GUI or AI models):

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r mcp_server/requirements.txt
```

Start the stdio server from the repository root:

```bash
start_mcp.bat
```

Exposed tools:

- `refine_pixel_art`
- `resize_image`
- `remove_image_background`
- `launch_desktop_app`
- `check_splitter_drag`
- `check_desktop_startup`
- `inspect_image`
- `check_project_health`
- `remove_fake_checkerboard`

This is a stdio server: it waits for JSON-RPC input, has no browser page, and is
normally started and stopped by an MCP client. A blank terminal is expected.
Diagnostics go to stderr; stdout is reserved for protocol messages.

Tools accept local image paths and write results to `mcp_outputs/` under the
server's working directory. Currently each tool reuses its fixed output filename.
The background smoke test covers color removal; AI mode has not been verified
and needs further model-path integration and optional ONNX dependencies.

`inspect_image` reports mode, dimensions, and alpha coverage without modifying
the file, which is useful when diagnosing transparent sprite exports.
`remove_fake_checkerboard` also supports `cleanup_islands`,
`min_component_size` (default 4), and `edge_shrink` (default 1) to remove
isolated residual pixels and checker-coloured edge fringes while preserving
connected foreground details.
`check_project_health` compiles project Python files and imports the core modules
(and optionally the desktop module), making it safe to run before parallel
feature work. `check_desktop_startup` separately constructs the Qt main window
in offscreen mode.

Run the end-to-end smoke test from the repository root:

```powershell
.venv\Scripts\python.exe -m mcp_server.smoke_test
```

The test launches a server subprocess, checks initialize/tools/list/tools/call,
inspects generated PNG dimensions and transparency, checks invalid arguments,
then closes the server and removes temporary outputs. It also rejects diagnostic
text on stdout. It does not register this server with any desktop MCP client.

For an MCP client, use these stdio process settings (adjust the checkout path):

```json
{
  "command": "E:/Project/TestProject/PerfectPixelTool/.venv/Scripts/python.exe",
  "args": ["-u", "E:/Project/TestProject/PerfectPixelTool/mcp_server/server.py"],
  "cwd": "E:/Project/TestProject/PerfectPixelTool"
}
```

The dependency is constrained to SDK v1 for the existing FastMCP API; see the
[official SDK's version guidance](https://github.com/modelcontextprotocol/python-sdk).
