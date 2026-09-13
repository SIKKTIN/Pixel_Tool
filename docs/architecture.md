# Perfect Pixel Tool 架构说明

## 当前分层

```text
入口层
├─ desktop_app.py       PySide6 主窗口与 Tab 组装
├─ app.py               Gradio/Web 入口
├─ start_mcp.bat        MCP stdio 启动脚本
└─ integrations/comfyui ComfyUI 节点入口

功能层
├─ image_crop.py        裁切与自由轮廓 UI
├─ image_resizer.py     尺寸缩放 UI
├─ image_splitter.py    图像切割 UI
├─ manual_editor.py     手动编辑 UI
└─ sequence_preview.py  序列帧预览 UI

算法层（src/）
├─ perfect_pixel         像素网格检测、采样、缩放、去背景
└─ watermark_remover     SLBR/LaMa 去水印

服务层
└─ mcp_server            图片处理与桌面启动检查的 MCP stdio 服务
```

## 目前需要留意的边界

- `desktop_app.py` 同时承担应用组装、暂存区、图像转换、线程和多个 Tab 的实现，文件较大。新的功能应优先放在独立模块，再由主窗口组装。
- `image_crop.py` 和 `watermark_remover/widget.py` 对 `desktop_app.ImageBuffer` 有直接或间接依赖，存在循环导入和运行时注入依赖。后续应将 `ImageBuffer` 提取到独立的核心模块，并通过构造函数或协议传入。
- 算法模块不应导入 PySide6，也不应依赖当前窗口状态。算法输入输出建议统一为 `numpy.ndarray`，并明确 RGB/RGBA 约定。
- MCP 服务当前是文件路径接口，适合自动化回归；它不替代桌面 UI 的鼠标交互测试。GUI 变更至少应配合 `check_desktop_startup` 和独立的构造测试。

## 建议的渐进式重构顺序

1. 新建 `src/perfect_pixel/app_core/image_buffer.py`，迁移 `ImageBuffer`，保留 `desktop_app.ImageBuffer` 兼容导出。
2. 新建 `src/perfect_pixel/app_core/image_io.py`，集中 PNG/RGBA 读写、棋盘格预览转换和透明度清理。
3. 将每个 Tab 保持为独立 QWidget 模块；`desktop_app.py` 只保留 `MainWindow`、依赖组装和入口。
4. 为 MCP 增加输入格式检查、RGBA 回归测试和唯一输出文件名，避免并行调用互相覆盖。
5. 保留顶层脚本作为薄兼容入口，优先通过 `python -m ...` 或 `pyproject.toml` 脚本运行。

## 并行开发约定

- 算法、桌面 UI、MCP、文档分别在独立分支或 worktree 修改。
- 共享接口变更先更新文档和最小 smoke test，再修改调用方。
- GUI 任务合并前运行 `check_desktop_startup`；算法任务运行透明度、尺寸和 RGB/RGBA 回归测试。
- 不在多个分支同时重排 `desktop_app.py`；先完成公共核心模块迁移，再拆分窗口文件。

