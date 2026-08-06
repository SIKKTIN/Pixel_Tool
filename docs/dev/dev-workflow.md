# 开发循环手册（dev.bat + 手动验证 + 调试技巧）

> 面向**日常改代码**的开发者：怎么改、怎么跑、怎么验证、怎么调试。
> 打包相关见 [`packaging.md`](packaging.md)。

---

## 1. 一句话开发循环

```
编辑代码 → Ctrl+C 停止 dev.bat → 再跑 dev.bat → 肉眼看效果
```

> ⚠️ **没有热重载**。PySide6 没有像 Flask 那种 `app.run(debug=True)` 自动重载能力，每次都要手动重启。这不是 bug，是 Qt 的现实。

---

## 2. 为什么是 `dev.bat`

| 入口 | 何时用 |
| :--- | :--- |
| [`dev.bat`](../../dev.bat) | **日常默认**。直接 `python desktop_app.py`，带控制台可见 `print` / `traceback` |
| `desktop_app.py` 直接跑 | 等价于 `dev.bat`（手动跑时自动应用 model 路径） |
| `pythonw desktop_app.py` | 想看 UI 但不想要黑窗时（**会丢失所有 print 输出**，调试不推荐） |
| `python app.py` | **只想调试核心算法**（像素细化），不想开 6 个 Tab 时。Gradio Web 界面 7860 端口 |
| `python example.py` | 纯算法 REPL 验证，无 GUI，最快（≈ 1 秒看到 `output_*.png`） |

`dev.bat` 比直接 `python desktop_app.py` 多做的事只有一件：

```bat
REM 默认指向本地 models/，没找到就退回 ..\Test\src\models\
set PERFECTPIXEL_MODEL_DIR=%~dp0models
if not exist "%PERFECTPIXEL_MODEL_DIR%\big-lama.pt" (
    if exist "..\Test\src\models\big-lama.pt" (
        set PERFECTPIXEL_MODEL_DIR=..\Test\src\models
    )
)
```

详见 [§4 模型放置策略](#4-模型放置策略)。**第一次跑**会用到这个 fallback；项目带了自己的模型后，`PERFECTPIXEL_MODEL_DIR` 就指 `models/` 本地路径。

---

## 3. 标准循环（3 个动作）

```bat
:: 项目根目录
cd e:\Project\TestProject\PerfectPixelTool

:: [1] 启动
dev.bat

:: [2] 改了代码 → 回到控制台窗口 Ctrl+C 停止（dev.bat 会 pause）

:: [3] 再跑一次
dev.bat
```

**首次启动**：弹出主窗口默认在「🎨 像素细化」Tab。

**每次改动后的验证清单**（30 秒过一遍）：

| Tab | 验证动作 | 预期 |
| :--- | :--- | :--- |
| 任意 | 拖一张 PNG 到窗口 | 主区域显示 |
| 像素细化 | 点「生成像素图」 | 控制台打印 grid size + 输出新图 |
| 暂存区 | 点「加入暂存区」 | 右侧出现缩略图，双击可送回主区 |
| 跨 Tab | 在 A Tab push → 切到 B Tab → 从暂存区 load | 图片流转通 |

---

## 4. 模型放置策略

项目里能放模型的地方有 3 个：

```
E:\Project\TestProject\
├── Test\src\models\                         ← 原始仓库（共享给多个项目）
│   ├── big-lama.pt
│   └── slbr.pth.tar
└── PerfectPixelTool\models\                 ← 项目本地副本（推荐）
    ├── big-lama.pt
    └── slbr.pth.tar
    └── isnet.onnx                           ← 去背景用
```

**查找优先级**（`lama_model.py:_env_model_dir` + `_exe_dir`）：

```
1. 环境变量 PERFECTPIXEL_MODEL_DIR 覆盖（dev.bat 设置）
2. EXE 同目录 / models/                       ← 打包后
3. 脚本 __file__ 父目录 / models/             ← dev 模式 + 本地 models/
4. ..\Test\src\models\                       ← dev.bat 的 fallback
5. 打包后同目录 EXE/../models/
```

**建议**：
- 平时 dev：把模型复制到 `PerfectPixelTool\models\` 一次，从此 `dev.bat` 不再需要 `..\Test\` 项目
- `Test\` 项目不必保留（除非你同时维护它）
- 模型不进 git（`.gitignore` 已忽略 `.onnx` / `.pt` / `.pth`），首次 clone 后手动放

---

## 5. 验证手段（没有 pytest 框架）

**这是个没有自动化测试的项目**。验证全靠手动：

### 5.1 算法层面（最快）

```bash
# 修改 src/perfect_pixel/perfect_pixel.py 后,秒级验证
python example.py
# 看 output_*.png 是否符合预期
```

`example.py` 默认读 `images/girl.jpg`，注释里列了 6 张备选测试图。

### 5.2 单 Tab 层面（中速）

```bash
python app.py    # Gradio 界面,只覆盖像素细化
# 浏览器 http://127.0.0.1:7860
```

适合纯算法 / 单 Tab 修改，避免 6 Tab 全打开的分心。

### 5.3 全栈 GUI（最贴近用户）

```bash
dev.bat
# 手动跑完 [§3 验证清单]
```

任何改 `desktop_app.py` / `manual_editor.py` / `image_splitter.py` 的提交都应该走这一步。

### 5.4「伪测试」:一次性脚本

如果想验证一段代码但不想进 GUI，写个 `scratch.py` 跑完即扔：

```python
# scratch.py — 随手验证片段,用完删
import numpy as np
from perfect_pixel import get_perfect_pixel
img = np.zeros((64, 64, 3), dtype=np.uint8)
img[::8, ::8] = 255    # 64x64, 8 像素网格
w, h, out = get_perfect_pixel(img)
assert (w, h) == (8, 8), f"got ({w},{h})"
print("OK")
```

`scratch.py` 不应入 git。

---

## 6. 调试技巧

### 6.1 print 是主力

dev.bat 用的是 `python`（非 `pythonw`），所以所有 `print()` 直接进控制台。配合：

```python
# desktop_app.py 里随便哪加
print(f"[DEBUG] grid_size detected: {w}x{h}, refine took {t1-t0:.2f}s")
```

关闭窗口后控制台还能看到最后几行。

### 6.2 PySide6 断点

用 `pdb` 嵌入 GUI 程序时要注意：**普通 `pdb.set_trace()` 会卡死 Qt 事件循环**。

```python
# 推荐:用 QTimer 单步触发,不阻塞事件循环
from PySide6.QtCore import QTimer
QTimer.singleShot(0, lambda: print("next event tick"))
# 或者更狠的:把断点设成条件断点
import pdb
pdb.set_trace()   # OK,因为 console 是 command loop,Qt 在另一个线程
```

更现代的做法：`pip install ptvsd` / VSCode 的 debugpy 附加。

### 6.3 找谁在吃内存

```python
# 加到 ImageBuffer.__init__
import psutil
def mem_mb(): return psutil.Process().open_files()
print(f"[MEM] {len(mem_mb())} open files, RSS={psutil.Process().memory_info().rss/1024/1024:.1f} MB")
```

PySide6 + torch 的常见内存杀手：
- 暂存区 `ImageBuffer._max_items` 设太大（默认 20）—— 改小或加 LRU 淘汰
- QtPixmap 不释放 —— `del pixmap; gc.collect()`
- torch 模型加载两次 —— 检查 `WatermarkWidget` 的 `try/except` 是否在 finally 里 unload

### 6.4 看 Qt 的事件流

```python
# desktop_app.py 顶部加,会打印所有 mousePress / paint 事件
from PySide6.QtCore import qInstallMessageHandler
def qt_msg(mode, ctx, msg): print(f"[Qt] {msg}")
qInstallMessageHandler(qt_msg)
```

适合查「为什么按钮点不下去」「为什么没重绘」这类 UI bug。

---

## 7. PySide6 没有热重载，怎么办

Qt 应用无法 `import reload`，所以加新 Tab 或大改 `MainWindow` 必然要重启。**两个加速技巧**：

### 7.1 import 局部化

新 Tab 模块顶层 import 耗时的大依赖（torch / onnx）放函数内部：

```python
# image_splitter.py
class ImageSplitterWidget:
    def __init__(self):
        # 不要在 import 时就 import torch
        pass

    def on_run(self):
        from torch import ...     # 第一次点按钮时才 import
        ...
```

效果：`desktop_app.py` 启动时间不会被单 Tab 拖累。

### 7.2 启动时间基线

实测参考：
- 干净启动（无 torch）: ≈ 2 秒
- 启动 + 加载 torch: ≈ 8 秒
- 启动 + 加载 torch + 加载 2 个模型权重: ≈ 15 秒

如果启动突然变慢，加 `import time; t0=time.time()` 到 `MainWindow.__init__` 各阶段定位。

---

## 8. ImageBuffer 协议（加新 Tab 前必读）

所有 Tab 之间不直接调用，通过中央 `ImageBuffer` 传图。

```python
from desktop_app import image_buffer     # ← 全局唯一实例

buf = image_buffer()                     # 拿到单例
buf.push(image_np, source_tab="A", source_file="x.png")   # 推入
img = buf.get_by_id(item_id)             # 按 ID 取
img = buf.current()                       # 取最近 push 的
```

设计约定：
- `push` 后必须能 `get_by_id` 找到
- 推入前 `image` 是 `np.ndarray`，shape `(H, W, 3)` 或 `(H, W, 4)`，`dtype=uint8`
- `source_tab` 是字符串标签，便于 UI 显示从哪里来的
- Tab 内部如有多个步骤，建议 push 一个最终结果，避免暂存区被半成品污染

**新 Tab 接入模式**（参考 `image_splitter.py`）：

```python
# 你的新 Tab 模块: my_new_tab.py
from desktop_app import ImageView, ImageBuffer   # 复用现有控件

class MyNewWidget(QWidget):
    def __init__(self):
        super().__init__()
        # ... 自己的 UI ...
        self._buf_ref: ImageBuffer | None = None

    @classmethod
    def set_buffer_ref(cls, buf: ImageBuffer):
        cls._buf_ref = buf

    def load_from_buffer(self, img: np.ndarray):
        """被暂存区双击触发,主区域载入图片"""
        self._current = img
        self.update_preview()

    def on_open(self):     # 工具栏「打开」触发
        path, _ = QFileDialog.getOpenFileName(...)
        img = cv2.imread(path)
        self.load_from_buffer(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def on_save(self):     # 工具栏「保存」触发
        path, _ = QFileDialog.getSaveFileName(...)
        cv2.imwrite(path, cv2.cvtColor(self._current, cv2.COLOR_RGB2BGR))
```

注册到 `desktop_app.py` 的 `MainWindow.__init__`：

```python
from my_new_tab import MyNewWidget
MyNewWidget.set_buffer_ref(image_buffer())
self.my_tab = MyNewWidget()
self.tabs.addTab(self.my_tab, "🆕 新功能")
```

> ⚠️ 详见 [`packaging.md` §5](packaging.md#5-为什么这么多---hidden-import)：加了新 import 后要回 `build.bat` 加 `--hidden-import`。

---

## 9. 项目里的「调试残留文件」

根目录有一批**一次性**脚本 / 日志，跟发布无关，是开发时反复重构图标的副产物。**建议清理**（一次性 commit 删掉）：

| 文件 | 性质 | 建议 |
| :--- | :--- | :--- |
| `gen_ico.py` | 旧版 BMP-encoded ICO 生成器 | ⚠️ 保留以备重新生成图标 |
| `regen_ico.py` | 新版 PNG-in-ICO 生成器 | ⚠️ 保留，比 `gen_ico.py` 更通用 |
| `verify_ico.py` | 检查 ICO 每一层 | 可并入 `regen_ico.py` 末尾 |
| `check_icon.ps1` | 从 EXE 抽图标 | 一次性 |
| `icon_debug.log` | 日志 | 删（已被 `.gitignore *.log` 忽略） |
| `preview.py` | 4 行 ICO → PNG 转码 | 一次性 |
| `build_run*.log` (13 个) | 打包调试日志 | 删（已被 `.gitignore *.log` 忽略，但旧文件已入库） |
| `build_stdout.log` / `build_stderr.log` | 同上 | 删 |
| `output.png` / `output_8x.png` / `compare.png` | `example.py` 产物 | 已在 `.gitignore`？—— **没有！** 建议加 `output*.png compare*.png` |
| `assets/图标_clean_iter_iter_iter.png` | 中文文件名 + 多次改名痕迹 | **重命名为 `assets/app_icon_src.png`**（已经存在但未使用，猜测是替代品） |
| `dist/` / `build/` / `__pycache__/` / `*.egg-info/` | 编译产物 | 已在 `.gitignore` |

> 命名建议：把图标相关的 4 个脚本 + 1 个 PNG 源收进 `tools/icon/`：
>
> ```
> tools/icon/
> ├── regen_ico.py        ← 主入口
> ├── verify_ico.py       ← 检查用
> ├── check_exe_icon.ps1  ← 验证打包后的 EXE
> └── src.png             ← 重命名后的图标源
> ```

---

## 10. IDE 配置建议

### 10.1 VSCode

仓库根有 `Moonshine-Image-win32-x64.code-workspace`，但那是 **Moonshine Image 项目的**（不是 perfect-pixel 的）。如果想给本项目新建：

```jsonc
// .vscode/settings.json (建议)
{
    "python.analysis.extraPaths": ["src", "src/watermark_remover/slbr_runtime"],
    "python.analysis.autoImportCompletions": true,
    "python.defaultInterpreterPath": "venv/Scripts/python.exe",
    "terminal.integrated.cwd": "${workspaceFolder}"
}
```

> 这俩 `extraPaths` 就是 `build.bat --paths=` 的镜像 —— PYTHONPATH 不设的话 Pylance 找不到 `perfect_pixel` / `src.networks`。

### 10.2 PyCharm

`Settings → Project → Project Structure → Add Content Root`：
- `src/`
- `src/watermark_remover/slbr_runtime/`

否则 `from perfect_pixel import ...` 会标红。

---

## 11. 加新 Tab 的标准动作清单

按顺序做完 6 步：

1. **写模块** `my_tab.py`，继承 QWidget，提供 `load_from_buffer` / `on_open` / `on_save` 三个方法
2. **注册到 `desktop_app.py`** 的 `MainWindow.__init__` —— 模仿 `image_splitter.py` 的 4 行（import / set_buffer_ref / 实例化 / addTab）
3. **跑一次** `dev.bat`，确认 Tab 出现、可以打开图、处理
4. **打包验证** —— 在 `build.bat` 的 `--hidden-import` 段补你的新模块依赖（详见 [`packaging.md` §5](packaging.md#5-为什么这么多---hidden-import)）
5. **打包测试** `build.bat`，双击 EXE 确认 Tab 仍正常
6. **写文档** —— 在 `docs/03-desktop-app.md` 加一节介绍新 Tab（待写）

---

## 12. 常见开发坑

### Q1: 改了 `src/perfect_pixel/`，但运行时还是老逻辑
检查 `__pycache__/` 是否被删：`desktop_app.py` 重新 import 时如果 `.pyc` 时间戳对得上，**Python 不会重新编译**。手动 `find . -name __pycache__ -exec rm -rf {} +` 或重启 IDE。

### Q2: 启动时报 `ModuleNotFoundError: No module named 'src.networks'`
`dev.bat` 没设 PYTHONTHON，所以 SLBR 找不到自己的 `src.*` 命名空间。临时方案：

```bat
:: dev.bat 顶部加
set PYTHONPATH=%~dp0src;%~dp0src\watermark_remover\slbr_runtime;%PYTHONPATH%
```

### Q3: 改了 `desktop_app.py` 出现奇怪 bug，最快的恢复办法
- `Ctrl+Z` 撤销
- 或 `git diff desktop_app.py` 看改了什么
- 或临时 `git stash` 后回干净状态排查

### Q4: image_splitter.py / manual_editor.py 里的 `from desktop_app import ImageView, ImageBuffer` 循环 import 风险
PySide6 应用一般不会循环，但加了新 Tab 后别忘了：

```python
# my_tab.py
def load_from_buffer(self, img):
    ...
```

并在主窗口注册时 `set_buffer_ref(buf)`。

### Q5: Gradio vs PySide6，两套前端同步更新？
**目前不强制同步**。`app.py` 只覆盖像素细化这一个工具，其他 5 个 Tab 只在桌面端有。如果你想在 Web 端也加 Tab，要单独写 Gradio Blocks（详见 `app.py:72` 的结构模仿）。

---

## 13. 相关文件清单

| 路径 | 用途 |
| :--- | :--- |
| [`dev.bat`](../../dev.bat) | 一键开发模式（推荐入口） |
| [`desktop_app.py`](../../desktop_app.py) | 主入口 |
| [`app.py`](../../app.py) | Gradio Web UI（单 Tab） |
| [`example.py`](../../example.py) | 算法 REPL 验证 |
| [`image_splitter.py`](../../image_splitter.py) | 新 Tab 接入范本 |
| [`manual_editor.py`](../../manual_editor.py) | 另一份 Tab 范本（RGBA 画笔） |
| [`requirements.txt`](../../requirements.txt) | dev 时装的依赖 |
| `models/.cache/huggingface/` | SLBR 评估时可能产生的 HF 缓存（无害） |

---

## 14. 改进方向（待办）

- [ ] 引入 `pytest` + `pytest-qt`，把 5 张测试图封装成 fixture，至少给 `get_perfect_pixel` 加 10 个用例
- [ ] 写 `scratch/` 目录规范：放一次性验证脚本，`.gitignore` 整目录
- [ ] 收敛调试残留文件（[§9](#9-项目里的调试残留文件)）到一个 `tools/icon/`
- [ ] 给 `PERFECTPIXEL_MODEL_DIR` 路径优先级加单元测试
- [ ] VSCode workspace 文件按本项目新建（不复用 Moonshine-Image 的）