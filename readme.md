# Perfect Pixel Tool

Perfect Pixel Tool 是一个面向像素风素材的本地图像工具集，组合了像素网格检测、RGBA 保真处理、缩放、去水印、去背景、手动编辑、裁切、序列帧预览和 MCP 自动化接口。

## 功能

- 自动检测并规整像素网格（OpenCV / NumPy 后端）
- 保留 PNG 的 RGBA 和半透明边缘
- 最近邻缩放、去水印、去背景、手动编辑
- 矩形裁切和自由轮廓裁切，裁切外区域保持透明
- 序列帧预览、播放/暂停、FPS 调整和逐帧查看
- PySide6 桌面应用、Gradio 网页入口、ComfyUI 节点和 MCP stdio 服务

## 项目结构

```text
PerfectPixelTool/
├─ desktop_app.py             # PySide6 桌面入口和模块装配
├─ app.py                     # Gradio 网页入口
├─ image_crop.py              # 矩形/自由轮廓裁切
├─ image_resizer.py           # 最近邻缩放
├─ image_splitter.py          # 图片切割
├─ manual_editor.py           # 手动编辑
├─ sequence_preview.py        # 序列帧预览
├─ src/perfect_pixel/         # 核心像素网格算法
├─ src/watermark_remover/     # SLBR/LaMa 去水印
├─ src/perfect_pixel/background_remover.py # 颜色、通道和 ONNX 去背景
├─ integrations/comfyui/      # ComfyUI 集成
├─ mcp_server/                # MCP stdio 服务和 smoke test
├─ models/                    # 可选模型文件
├─ dev.bat                    # Windows 开发启动
├─ build.bat                  # PyInstaller 构建
└─ requirements.txt           # 桌面/算法依赖
```

核心算法通过 `src/perfect_pixel/__init__.py` 选择后端：安装 OpenCV 时使用 `perfect_pixel.py`，否则回退到 `perfect_pixel_noCV2.py`。桌面层负责界面和模块间的图像缓冲，算法层不依赖 Qt。

## 安装与启动

推荐 Python 3.10+：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.\dev.bat
```

也可以直接运行 `.venv\Scripts\python.exe desktop_app.py`。若提示缺少依赖：

```powershell
.venv\Scripts\python.exe -m pip install numpy opencv-python Pillow PySide6
```

网页入口：`.venv\Scripts\python.exe app.py`。

## MCP 服务

MCP 服务位于 `mcp_server/server.py`，采用 stdio JSON-RPC；stdout 只用于协议消息，诊断信息写入 stderr。启动：

```powershell
.\start_mcp.bat
```

当前工具：

| 工具 | 用途 |
| --- | --- |
| `refine_pixel_art` | 自动检测网格并细化像素图 |
| `resize_image` | 最近邻缩放 |
| `remove_image_background` | 颜色/通道/可选 AI 去背景 |
| `check_desktop_startup` | offscreen 创建 `QApplication` 和 `MainWindow`，捕获启动异常 |
| `inspect_image` | 检查图片模式、尺寸和 Alpha 统计 |
| `check_project_health` | 编译 Python 文件并检查核心模块导入 |

客户端配置：

```json
{
  "command": "E:/Project/TestProject/PerfectPixelTool/.venv/Scripts/python.exe",
  "args": ["-u", "E:/Project/TestProject/PerfectPixelTool/mcp_server/server.py"],
  "cwd": "E:/Project/TestProject/PerfectPixelTool"
}
```

运行端到端检查：

```powershell
.venv\Scripts\python.exe -m mcp_server.smoke_test
```

输出文件默认写入 `mcp_outputs/`。协议和依赖说明见 [`mcp_server/README.md`](mcp_server/README.md)。

## 开发与验证

提交前至少执行：

```powershell
.venv\Scripts\python.exe -m compileall desktop_app.py image_crop.py image_resizer.py image_splitter.py manual_editor.py sequence_preview.py src mcp_server
.venv\Scripts\python.exe -m mcp_server.smoke_test
```

涉及桌面初始化的改动，再运行 `check_desktop_startup`。涉及透明度的改动，确认输入和输出均为 RGBA，并检查完全透明像素的 Alpha 为 0。

## 并行开发约定

1. 算法任务只改 `src/`，通过公开函数和样例图片验证。
2. 桌面功能任务优先改对应模块；需要改 `desktop_app.py` 时先约定初始化和信号连接位置。
3. MCP 任务只改 `mcp_server/`，新增工具必须补 schema、错误处理和 smoke test。
4. 文档和构建任务分别改 `readme.md`、`docs/`、`build.bat`，不要提交临时输出。
5. 每个并行任务完成后运行语法检查；合并前运行完整 MCP smoke test 和桌面启动检查。

模块通过稳定接口协作：图像数组使用 `numpy.ndarray`，颜色通道明确标注 RGB/RGBA，桌面模块通过 `ImageBuffer` 交换结果。新增功能应先定义输入、输出和 Alpha 行为，再接入 UI。

## 构建发布

```powershell
.\build.bat
```

构建配置位于 `PerfectPixelTool.spec`。模型文件较大时不要复制进源码提交，使用 `models/` 下的本地路径或单独配置。

## 许可证

项目沿用 MIT 许可证；第三方模型和依赖请遵循各自许可证。
