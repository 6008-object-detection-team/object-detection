# 来源与贡献

本项目为 [6008-object-detection-team/object-detection](https://github.com/6008-object-detection-team/object-detection) 的团队检测应用。2026-10-07 公开整合以团队 `main` 的 `bd298cd74154b3ea1e46182f73080d97f110ee79` 为基础，保留该仓库公开历史。

本轮整合来源包括：

- 团队已有图片输入、ByteTrack 目标编号、CUDA 安装与暗光处理。
- Pan Deng 的 Grounding DINO Tiny 接入、逐类别 token 打分修正与 `pan_deng/` 标注评测工具。
- 经用户确认可公开的 pd / `pdsky` 升级工作：SAVPE 视觉提示、SAHI、资源保护、小型 COCO 数据、Windows 适配、摄像头最新帧调度与录像时间轴。
- 本机合并兼容、Windows 测试夹具修复、依赖调整与验证脚本。

仅整理源码、说明和必要测试资料；私人仓库保持私人，未导入其 Git 提交历史或原始实验记录。历史模型 / 构建文件仍属于原公共仓库，未在本次重新上传。

该说明不授予新的软件、权重或图片许可。第三方软件与模型遵守各自许可；COCO 人工标注及逐图来源 / 许可保留在 `datasets/*/manifest.json`。模型固定下载 URL、大小与 SHA-256 见 `model_sources.json`。
