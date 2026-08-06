# docs/dev/ — 开发文档

> 面向**项目维护者 / 贡献者**的文档：构建、打包、调试、性能调优、安全发布等。

## 当前文档

| 文档 | 用途 |
| :--- | :--- |
| [packaging.md](packaging.md) | 基于 `build.bat` 的 PyInstaller 打包指南，包含产物布局、模型 sidecar、故障排查 |
| [dev-workflow.md](dev-workflow.md) | 日常开发循环：编辑 / 运行 / 验证 / 调试 / 新增 Tab 范式、调试残留清单 |

## 计划待写

- [ ] `ci-cd.md` — 自动化构建、签名、发布（未来规划）
- [ ] `release-checklist.md` — 发版前自检清单
- [ ] `cross-platform.md` — 跨平台打包注意事项（macOS / Linux）

## 约定

- 命令均假设在项目根目录 `PerfectPixelTool/` 下执行
- Windows 优先（项目主战场），macOS / Linux 标注差异
- 所有路径用相对仓库根目录表示
