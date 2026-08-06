# 打包指南（PyInstaller · onedir）

> 把 `desktop_app.py` 打包成可在没有 Python 环境的 Windows 机器上直接运行的可执行文件。
> 本文档完全基于 [`build.bat`](../../build.bat) 的实际脚本内容，不是凭空推演。

---

## 1. 产物形态

`build.bat` 使用 **`--onedir`** 模式（非 `--onefile`），输出一个完整的应用目录：

```
dist/PerfectPixelTool/
├── PerfectPixelTool.exe        ← 主可执行（≈ 6 MB，纯启动器）
├── _internal/                  ← PyInstaller 私有运行时（≈ 1.5 GB，含 torch）
│   ├── python311.dll
│   ├── PySide6/...
│   ├── torch/...
│   └── ...（几百个 .pyd / .dll）
├── assets/
│   ├── app_icon.ico
│   └── models/                 ← ONNX 模型（去背景 Tab 用）
└── models/                     ← Torch 模型（去水印 Tab 用）
    ├── big-lama.pt             ← --add-data 没带，build.bat 第 5 步从外部拷贝
    └── slbr.pth.tar
```

**必须把整个 `dist/PerfectPixelTool/` 目录一起分发**，进入该目录双击 `PerfectPixelTool.exe` 运行。

---

## 2. 为什么是 onedir 而不是 onefile

`build.bat` 注释里写得很清楚：

> `--onedir` 模式（vs `--onefile`）产生标准应用目录结构，**避免单文件 EXE 写入时与 Windows 索引器的锁冲突**。
> 运行前请进入 `dist\PerfectPixelTool\` 双击 EXE。

实践经验：
- `--onefile` 启动时会先把文件解压到 `%TEMP%`，体积大时解压慢且被某些杀软误报
- `--onedir` 启动秒开，目录摆在那儿也好排查
- 代价：分发时是文件夹不是单个 exe，要打个 zip

---

## 3. 打包流程

### 3.1 一键方式（推荐）

```bat
:: 项目根目录
cd e:\Project\TestProject\PerfectPixelTool
build.bat
```

脚本会按 5 步执行：

| 步骤 | 内容 |
| :--- | :--- |
| 1/5 | 检查 Python 环境 |
| 2/5 | 检查 / 安装 PyInstaller、PySide6 |
| 3/5 | 清理 `build/` / `dist/` / `PerfectPixelTool.spec` |
| 4/5 | 调用 PyInstaller 打 onedir（耗时 3~8 分钟，含 torch） |
| 5/5 | 把 torch 模型从 `..\Test\src\models\` 复制到 `dist\PerfectPixelTool\models\` |

打包结束后会问 `是否现在启动测试? [y/N]`，输入 `y` 可以直接打开 EXE 验。

### 3.2 手动方式（排查问题时用）

```bat
python -m PyInstaller ^
    --noconsole ^
    --onedir ^
    --windowed ^
    --name PerfectPixelTool ^
    --icon=assets\app_icon.ico ^
    --paths=src ^
    --paths=src/watermark_remover/slbr_runtime ^
    --add-data=assets/app_icon.ico;assets ^
    --add-data=assets/models;assets/models ^
    --hidden-import=...                 ← 见第 5 节
    desktop_app.py
```

> ⚠️ **必须用 `^` 续行**（PowerShell 用反引号 `` ` ``，且 PowerShell 对 `--add-data` 的 `;` 需要转义）。建议直接用 `build.bat`。

---

## 4. 关键参数逐项解释

| 参数 | 作用 | 没了会怎样 |
| :--- | :--- | :--- |
| `--noconsole` | 启动时不弹黑色控制台窗口 | 跑 EXE 会同时弹一个黑窗，玩家体验差 |
| `--onedir` | 产出一个完整目录（vs 单文件） | 见第 2 节 |
| `--windowed` | GUI 子系统（Windows 专用） | 同上 |
| `--name PerfectPixelTool` | 输出目录 / EXE 名字 | 默认叫 `desktop_app`，丑 |
| `--icon=assets\app_icon.ico` | 任务栏 / 资源管理器图标 | 用默认 Python 图标 |
| `--paths=src` | 让 `from perfect_pixel import ...` 找得到 | **`ModuleNotFoundError: perfect_pixel`**（关键！） |
| `--paths=src/watermark_remover/slbr_runtime` | 让 SLBR 自带的 `src.*` 模块找得到 | 去水印模型加载失败 |
| `--add-data=assets/app_icon.ico;assets` | 把图标打进 EXE 内的 `assets/` | 运行时找不到图标 |
| `--add-data=assets/models;assets/models` | 把 ONNX 模型一起打包 | 去背景 AI 模式不可用 |
| `--hidden-import=...` | 强制包含 PyInstaller 静态分析没发现但运行需要的模块 | 一打开 EXE 就 `ImportError`（详见第 5 节） |

> `--paths=src` 是关键中的关键：`pyproject.toml` 里 `package-dir = {"" = "src"}` 让 `from perfect_pixel` 路径生效，PyInstaller 默认根本不知道这事，必须显式告诉它。

---

## 5. 为什么这么多 `--hidden-import`

PyInstaller 是**静态分析** `desktop_app.py` 的 import 来决定打包哪些模块。下面的情况会漏，需要手动补：

### 5.1 SLBR 模型运行时（13 个）

`src/watermark_remover/slbr_runtime/src/` 是 vendored 第三方代码，使用 `from src.networks import ...` 这种**非常规**的导入路径，PyInstaller 的静态分析扫不到：

```
--hidden-import=src.networks.resunet
--hidden-import=src.networks.blocks
--hidden-import=src.networks.discriminator
--hidden-import=src.networks.methods
--hidden-import=src.models.SLBR
--hidden-import=src.models.BasicModel
--hidden-import=src.utils.model_init
--hidden-import=src.utils.osutils
--hidden-import=src.utils.imutils
--hidden-import=src.utils.parallel
--hidden-import=src.utils.losses
--hidden-import=src.utils.misc
--hidden-import=src.utils.transforms
```

漏一个 → 一打开「去水印」Tab 就报错。

### 5.2 PyTorch 动态依赖

`torch` 大量使用 `__import__` + 反射，PyInstaller 抓不全：

```
--hidden-import=torch
--hidden-import=torch.nn
--hidden-import=torch.nn.functional
--hidden-import=torch.utils
--hidden-import=torch.utils.data
--hidden-import=torchvision
--hidden-import=torchvision.models
```

### 5.3 PySide6 子模块

只 import `PySide6.QtWidgets` 时，QtNetwork / QtMultimedia / QtPrintSupport 等不会被自动拉；为了支持以后扩展 Tab 里可能用到的拖拽 / 打印 / 多媒体，全列上：

```
--hidden-import=PySide6.QtCore
--hidden-import=PySide6.QtGui
--hidden-import=PySide6.QtWidgets
--hidden-import=PySide6.QtNetwork
--hidden-import=PySide6.QtMultimedia
--hidden-import=PySide6.QtMultimediaWidgets
--hidden-import=PySide6.QtOpenGL
--hidden-import=PySide6.QtPrintSupport
--hidden-import=PySide6.QtQml
--hidden-import=PySide6.QtQuick
--hidden-import=PySide6.QtSvg
--hidden-import=PySide6.QtWebEngineCore
--hidden-import=PySide6.QtWebEngineWidgets
```

> ⚠️ 注意 `--hidden-import=PySide6.QtWidgets` 出现**两次**（第 81 行和第 92 行），这是历史遗留，不影响功能，但删掉前请确认所有用到 QtWidgets 的地方都依赖别处的链入。

### 5.4 其他

```
--hidden-import=pytorch_ssim            ← SLBR 评估用
--hidden-import=pytorch_iou             ← SLBR 评估用
--hidden-import=cv2
--hidden-import=numpy
--hidden-import=onnxruntime              ← 去背景 AI 模式
--hidden-import=PIL
--hidden-import=PIL.Image
--hidden-import=PIL._tkinter_finder      ← Pillow 的可选 tkinter 后端
--hidden-import=perfect_pixel.perfect_pixel
--hidden-import=perfect_pixel.perfect_pixel_noCV2
```

---

## 6. 模型文件：双轨制

打包后有两类模型在两个地方：

| 模型 | 用途 | 打包方式 | 路径 |
| :--- | :--- | :--- | :--- |
| `isnet.onnx` | 去背景 AI 模式 | `--add-data=assets/models;assets/models` 打进 `_internal/` | `dist\PerfectPixelTool\_internal\assets\models\isnet.onnx` |
| `big-lama.pt` | LaMa 去水印 | **不打进 EXE**，build.bat 第 5 步从 `..\Test\src\models\` 复制 | `dist\PerfectPixelTool\models\big-lama.pt` |
| `slbr.pth.tar` | SLBR 去水印 | 同上 | `dist\PerfectPixelTool\models\slbr.pth.tar` |

**为什么 Torch 模型要 sidecar？**

- 单个 `big-lama.pt` ≈ 200 MB，`slbr.pth.tar` ≈ 50 MB —— 打进 EXE 会让 freeze 时间和首次启动解压时间都不可接受
- 模型经常更新（重新训练 / 换权重），单独放外面方便替换
- `build.bat` 自动从 `..\Test\src\models\` 复制，**前提是同级目录有 `Test/` 项目**。否则只看到 `[WARN]`，EXE 仍能跑，只是「去水印」Tab 报错

**如果不想依赖 `Test/` 项目**，手动复制即可：

```bat
copy big-lama.pt   dist\PerfectPixelTool\models\
copy slbr.pth.tar  dist\PerfectPixelTool\models\
```

---

## 7. 打包后自检

```bat
cd dist\PerfectPixelTool
.\PerfectPixelTool.exe
```

依次验证：

1. ✅ 主窗口正常打开，无黑窗
2. ✅ 6 个 Tab 都能切入（特别是「去水印」，确认没漏 `hidden-import`）
3. ✅ 拖一张 PNG 进去能渲染
4. ✅ 右侧暂存区能 push / pull
5. ✅ 「去背景」AI 模式能找到 `isnet.onnx`
6. ✅ 「去水印」能找到 `big-lama.pt` / `slbr.pth.tar`（如果放进去的话）

---

## 8. 常见问题

### Q1: 打包后 EXE 双击闪退

**症状**：一闪而过，没有任何窗口。

**排查**：
```bat
:: 临时改成有控制台模式，看报错
python -m PyInstaller --onedir --console --name PerfectPixelTool desktop_app.py
.\dist\PerfectPixelTool\PerfectPixelTool.exe
```
90% 是 `ModuleNotFoundError`，对照第 5 节补 `--hidden-import`。

### Q2: `ModuleNotFoundError: perfect_pixel`

说明缺少 `--paths=src`。这是因为 `pyproject.toml` 用 `package-dir = {"" = "src"}` 把源码放在非标准位置，PyInstaller 默认不会去找。

### Q3: 去水印 Tab 加载失败，去背景 Tab 正常

按 `desktop_app.py:2103` 的设计，此时不是 EXE 崩溃，而是显示一个「⚠️ 去水印模块加载失败」的占位面板。看面板里抛出的具体异常，对照第 5.1 节补 `src.*` 系列 hidden-import。

### Q4: 打包过程卡 10 分钟以上没动

正常。打包 torch 这步最慢，约 3~8 分钟。**不要中途 Ctrl+C**，EXE 目录会半截坏掉要重打。

### Q5: `_internal/` 太大（1.5 GB+）

主要原因：torch 本身就 800 MB+。
可选减肥手段（**不推荐现在做**，先确保能跑）：
- 上 [UPX](https://upx.github.io/) 压缩 `.pyd` / `.dll`
- 用 `--exclude-module` 砍掉用不到的 torch 子模块（如 `torchvision.io.video`）
- 拆分成多个 EXE（桌面 + Web 后端）

### Q6: 之前打包过，现在 rebuild 报错

build.bat 第 3 步会清理 `build/` / `dist/` / `*.spec`。如果手动跑 PyInstaller，记得同样清理。

### Q7: 怎么支持 macOS / Linux

`build.bat` 是 Windows 专用（含 `mkdir` / `copy` / `pause`）。其他平台写 shell 版：

```bash
pyinstaller --noconsole --onedir --windowed \
    --name PerfectPixelTool \
    --icon=assets/app_icon.icns \
    --paths=src \
    ... (其余 hidden-import 同) \
    desktop_app.py
```

⚠️ PySide6 在 macOS 上需要单独处理 `.app` bundle；Linux 缺一些 Qt 平台插件（`libxcb` 等）。详见 [cross-platform.md](cross-platform.md)（待写）。

---

## 9. 增量打包 / 调试技巧

| 场景 | 做法 |
| :--- | :--- |
| 只改了 `desktop_app.py` | 直接 `pyinstaller desktop_app.py`（不需传参数，会复用 `PerfectPixelTool.spec`） |
| 加了新 Tab / 新 import | 编辑 `build.bat` 加 `--hidden-import`，然后重打 |
| 只想看打包是否成功 | 打包完直接双击 EXE，看任务栏图标和窗口能否正常出现 |
| 改 icon 不想重打 | 单独替换 `assets/app_icon.ico`，再 `pyinstaller --icon=...` 只重打这一资源 |
| 编译缓存 | 删除 `%APPDATA%\pyinstaller` 可强制不缓存依赖分析结果 |

---

## 10. 相关文件清单

| 路径 | 用途 |
| :--- | :--- |
| [`build.bat`](../../build.bat) | 一键打包脚本（本指南的**唯一事实来源**） |
| [`dev.bat`](../../dev.bat) | 开发模式（不打包，直接 `python desktop_app.py`） |
| [`requirements.txt`](../../requirements.txt) | 桌面端运行时依赖 |
| [`pyproject.toml`](../../pyproject.toml) | 包定义，**不参与打包**（PyInstaller 不读它） |
| `PerfectPixelTool.spec` | PyInstaller 生成的中间配置，**不应入版本控制**（已在 `.gitignore`） |
| `build/` / `dist/` | PyInstaller 输出目录，**不应入版本控制** |

---

## 11. 改进方向（待办）

- [ ] 把 14+ 个 `--hidden-import` 收敛到一个 `--collect-submodules` 调用里
- [ ] 写一个 `pyinstaller --clean` 包装脚本，处理缓存
- [ ] 给 EXE 做代码签名（避免 SmartScreen 拦截）
- [ ] 写 GitHub Actions 自动打 tag → 自动构建 → 自动发 Release
- [ ] 写 `cross-platform.md` 覆盖 macOS `.app` bundle 和 Linux AppImage