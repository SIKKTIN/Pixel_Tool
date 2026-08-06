# 00 · 项目整体介绍

> **一句话**：一个本地化的「像素风图像处理」工具集 —— 既是一个可 `pip install` 的 Python 算法库，又是一个开箱即用的桌面 GUI 应用。

---

## 1. 项目定位

`perfect-pixel` 的核心使命是 **处理 AI 生成 / 二次采样的失真像素图**：

- AI（Stable Diffusion / ChatGPT / Gemini）生成的像素图常常是「看起来像像素但网格歪了」
- 普通升采样（双三次 / Lanczos）会进一步破坏网格
- 本工具自动 **检测原图的像素网格 → 对齐到规整方格 → 用中心 / 中位数 / 众数重新采样**，产出「真正像素级对齐」的图

围绕这个核心能力，又扩展出 5 个相关工具（缩放、去水印、去背景、手动编辑、图像切割），全部塞进同一个桌面应用里。

---

## 2. 两种使用形态

### 2.1 Python 库（算法层）

包名：`perfect-pixel`（PyPI）
入口模块：`perfect_pixel`
核心函数：`get_perfect_pixel(rgb, ...) -> (refined_w, refined_h, scaled_image)`

```bash
# 轻量版（仅 NumPy）
pip install perfect-pixel

# 推荐（OpenCV 加速）
pip install perfect-pixel[opencv]

# 全部可选依赖（含去水印的 PyTorch）
pip install perfect-pixel[all]
```

```python
import cv2
from perfect_pixel import get_perfect_pixel

bgr = cv2.imread("images/avatar.png")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
w, h, out = get_perfect_pixel(rgb)  # 自动检测网格 + 对齐 + 重采样
cv2.imwrite("refined.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
```

### 2.2 桌面应用（GUI 层）

入口：`desktop_app.py`
框架：PySide6（Qt6）
启动：

```bash
python desktop_app.py          # 开发模式
# 或直接双击打包好的 PerfectPixelTool.exe
```

应用主窗口是一个 6-Tab 的标签页 + 右侧图片暂存区。所有 Tab 共享中央 `ImageBuffer`：
- 任何 Tab 处理完一张图，可以**压入暂存区**
- 任何 Tab 可以**从暂存区取一张图**继续处理
- 形成「细化 → 缩放 → 去背景 → 手动修 → 切割」的流水线工作流

---

## 3. 桌面应用 6 个 Tab

| Tab | 入口类 | 功能 | 依赖 |
| :--- | :--- | :--- | :--- |
| 🎨 **像素细化** | `PixelRefineWidget` | 自动检测网格并对齐 | 核心库 |
| 📐 **尺寸缩放** | `ScaleWidget` | 按目标尺寸重新采样 | OpenCV |
| 🪄 **去水印** | `WatermarkWidget` | SLBR + LaMa 两个 AI 模型自动修复 | PyTorch（可选） |
| 🎭 **去背景** | `BackgroundRemoverWidget` | 颜色 / Alpha / AI 三种模式 | ISNet ONNX（可选） |
| ✏️ **手动编辑** | `ManualEditorWidget` | 画笔 + 橡皮擦精修 RGBA | 核心库 |
| ✂️ **图像切割** | `ImageSplitterWidget` | 按行 × 列网格切割，逐张可选入栈 | 核心库 |

> 🪄 Tab 如果加载失败会自动降级为一个「⚠️ 请安装 torch」的占位面板，不会让整个应用崩溃（见 `desktop_app.py:2103`）。

入口注册在 `desktop_app.py:2089-2137` 的 `MainWindow.__init__`。

---

## 4. 其他入口

| 入口 | 文件 | 说明 |
| :--- | :--- | :--- |
| Gradio Web 界面 | `app.py` | 只含「像素细化」一个工具，浏览器访问 `http://127.0.0.1:7860` |
| ComfyUI 节点 | `integrations/comfyui/` | 把核心算法封装成 ComfyUI 节点，可在工作流里调用 |
| 算法示例 | `example.py` | `cv2.imread` → `get_perfect_pixel` → `matplotlib` 可视化的最小例子 |

---

## 5. 技术栈

| 层 | 选型 | 说明 |
| :--- | :--- | :--- |
| 核心算法 | NumPy（强制）+ OpenCV（可选） | `__init__.py` 运行时优先选 OpenCV 版 |
| 桌面 GUI | PySide6 ≥ 6.5 | 单窗口多 Tab + 中央 ImageBuffer |
| 深度学习 | PyTorch ≥ 2.0 | 仅去水印 Tab 使用 |
| ONNX 推理 | onnxruntime | 去背景的 ISNet 模型 |
| Web UI（可选） | Gradio | `app.py` |
| 打包 | PyInstaller | `build.bat` / 手动命令 |
| Python | ≥ 3.9 | 见 `pyproject.toml` |

---

## 6. 顶层目录结构

```
PerfectPixelTool/
│
├── app.py                       # Gradio Web UI 入口
├── desktop_app.py               # 桌面应用主入口（PySide6，6 个 Tab）
├── example.py                   # 算法库最小使用例子
│
├── image_splitter.py            # ✂️ 图像切割 Tab（独立模块）
├── manual_editor.py             # ✏️ 手动编辑 Tab（独立模块）
│
├── pyproject.toml               # 包定义、入口点、可选依赖
├── requirements.txt             # 依赖锁定
├── readme.md                    # GitHub 用户向 README（英文）
│
├── src/                         # 真正的 Python 包
│   ├── perfect_pixel/           # ─── 核心算法（库）
│   │   ├── perfect_pixel.py            # OpenCV 版本
│   │   ├── perfect_pixel_noCV2.py      # 纯 NumPy 版本
│   │   └── __init__.py                 # 自动优先 OpenCV
│   │
│   └── watermark_remover/       # ─── 去水印（可选模块）
│       ├── lama_model.py
│       ├── slbr_runner.py
│       ├── widget.py                   # PySide6 Tab 控件
│       └── slbr_runtime/               # SLBR 模型运行时（vendored）
│
├── assets/                      # 文档用图、示例图、应用图标
├── images/                      # 测试图片
├── models/                      # AI 模型权重（isnet.onnx 等）
│
├── integrations/comfyui/        # ComfyUI 节点包
│
└── docs/                        # ← 你正在看的目录
```

> **几个容易混淆的点**
>
> 1. `desktop_app.py` 在仓库根目录，但它 import 的库在 `src/` —— 通过 `pyproject.toml` 的 `package-dir = {"" = "src"}` 让两者处于同一命名空间
> 2. `image_splitter.py` / `manual_editor.py` 之所以平铺在根目录而不是 `src/`，是因为它们**不属于要发布的库**，只服务桌面应用。打包时被 PyInstaller 一起带上即可
> 3. `src/watermark_remover/slbr_runtime/` 是 **vendored 的第三方代码**（SLBR 官方仓库），不要直接修改；如需升级请替换整个目录

---

## 7. 设计原则

1. **算法与 UI 解耦** — 核心算法在 `src/perfect_pixel/`，可独立被 Gradio / ComfyUI / 桌面 GUI 三种前端复用
2. **降级而非崩溃** — 可选模块（去水印）加载失败时显示占位面板，其他 Tab 仍可用
3. **中央暂存区串联 Tab** — `ImageBuffer` 是「胶水」，Tab 之间不直接调用，只通过 push / pull 交换图片
4. **库可独立发布** — `pyproject.toml` 把 `perfect_pixel` 和 `watermark_remover` 设为可独立安装的包，桌面应用是另一回事
5. **算法有 fallback** — `__init__.py` 在 OpenCV 缺失时自动切到纯 NumPy 实现，保证最低可运行性

---

## 8. 快速上手

### 跑桌面应用（最快）

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 启动
python desktop_app.py
```

打开后默认进入「🎨 像素细化」Tab，拖一张像素风图片到窗口即可看到效果。

### 跑算法库

```bash
python example.py
```

会读取 `images/girl.jpg`，生成 `output_*.png`。

### 跑 Web UI

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

### 打包桌面应用

```bash
pyinstaller --noconsole --windowed --onefile --name PerfectPixelTool desktop_app.py
# 产物在 dist/PerfectPixelTool.exe
```

---

## 9. 阅读建议

| 你的目标 | 先读 |
| :--- | :--- |
| 理解整个应用怎么跑起来 | [02 · 架构](architecture.md)（待写） |
| 改某个 Tab 的 UI | 对应 Tab 的源文件 + `desktop_app.py` 的注册段 |
| 改 / 优化核心算法 | `src/perfect_pixel/perfect_pixel.py` + [04 · 像素细化算法](algorithm-pixel-refine.md)（待写） |
| 加一个新 Tab | 模仿 `image_splitter.py` 的接入模式 |
| 集成去水印到自己的项目 | `src/watermark_remover/__init__.py` 看导出 |

---

## 10. 已知限制 / 待办

- `app.py`（Gradio 版）只覆盖了「像素细化」，未包含其他 5 个 Tab
- 「去水印」Tab 需要额外安装 PyTorch（≈ 800 MB），默认安装流程不会触发
- 「去背景」的 AI 模式需要本地存在 `models/isnet.onnx`
- 项目根目录有不少 **调试残留文件**（`build_run*.log`、`check_icon.ps1`、`verify_ico.py`、`preview.py` 等），下一步清理