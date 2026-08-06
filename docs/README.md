# PerfectPixelTool 文档

本目录是 **PerfectPixelTool / perfect-pixel** 的开发者文档入口。
面向想要读源码、改 Bug、加 Tab、或理解算法原理的读者。

> 如果你是终端用户想直接用工具，请直接看根目录的 [`README.md`](../readme.md)。

## 文档导航

| 文档 | 内容 |
| :--- | :--- |
| [00 · 项目整体介绍](overview.md) | 一句话定位、两种使用形态、6 个 Tab 概览、技术栈、目录结构 |
| 01 · 快速上手 | _（待写）_ 安装、运行、打包 |
| 02 · 架构 & 数据流 | _（待写）_ 中央 `ImageBuffer` 暂存区、Tab 通信协议、生命周期 |
| 03 · 桌面应用 Tab 参考 | _（待写）_ 每个 Tab 的输入 / 输出 / 可配置项 |
| 04 · 核心算法 · 像素细化 | _（待写）_ FFT 网格检测 + Sobel 边缘对齐的算法流程 |
| 05 · 去水印模块 | _（待写）_ SLBR + LaMa 双模型选型与切换 |
| 06 · 去背景模块 | _（待写）_ 三种模式（颜色 / Alpha / AI）实现细节 |
| 07 · 手动编辑模块 | _（待写）_ 画笔 / 橡皮擦 / Undo-Redo 实现 |
| 08 · 图像切割模块 | _（待写）_ 网格切割、可选中预览 |
| 09 · ComfyUI 集成 | _（待写）_ `integrations/comfyui/` 节点包说明 |
| 10 · 打包 & 发布 | _（待写）_ PyInstaller 打包脚本、产物布局 |

## 开发者文档 → [`dev/`](dev/README.md)

> 面向**项目维护者 / 贡献者**的文档：构建、打包、调试、发布。
>
> | 文档 | 内容 |
> | :--- | :--- |
> | [dev/packaging.md](dev/packaging.md) | 基于 `build.bat` 的 PyInstaller 打包指南（onedir 模式、hidden-import 详解、模型 sidecar、故障排查） |
> | [dev/dev-workflow.md](dev/dev-workflow.md) | 日常开发循环（dev.bat / 模型放置 / 手动验证 / PySide6 调试 / 加新 Tab 范式 / 调试残留清单） |
> | dev/release-checklist.md | _（待写）_ 发版前自检清单 |
> | dev/cross-platform.md | _（待写）_ macOS / Linux 打包差异 |

## 贡献指引

- 改 Tab 控件 → 阅读 `overview.md` 了解 6 个 Tab 的职责分工，再读 `02 · 架构`
- 改核心算法 → 阅读 `04 · 像素细化算法`
- 加新 Tab → 参考 `image_splitter.py` 的接入方式（`set_buffer_ref` + `addTab`）

## 文档约定

- 所有文件路径均相对仓库根目录 `PerfectPixelTool/`
- 代码引用使用 `path/to/file.py:行号` 形式
- 截图 / 动图统一放在 `docs/assets/`