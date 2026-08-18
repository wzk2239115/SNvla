# AGENTS.md — 工作约定

## 命令执行偏好

- **所有命令都前台运行，不要用 `nohup ... &`、`tmux`、`screen` 等后台方式**
  给出的示例命令直接前台跑，用户自己决定怎么调度。

## 远程机器

- 8 卡机：`nb-wangzekai-ctm-01-0`（8×H100）
- 4 卡机：`nb-wangzekai-ctm-02-4-0`（4×H800）
- 数据：`/home/jovyan/exploitgym/D2E-Original`（1.7TB, 459 episodes, 29 games）
- 帧缓存：`/home/jovyan/exploitgym/frame_cache`（stride=8, 224px JPEG）
- Mage-VL：`/home/jovyan/exploitgym/Mage-VL`
- 强 LLM：`/home/jovyan/h800fast/wangzekai/Qwen3.6-27B`（4卡机上, vLLM serve 用）
- 内网代理：`export http_proxy=http://public-proxy.qihoo.net:3128 https_proxy=...`

## 仓库约定

- `assets/` 目录（D2E_sample、Mage-VL、ocap、论文、plan.md）**永远不提交**
- `tmp/`、`probes/`、`checkpoints/`、`index_cache/` 不提交
- 提交前检查 `git status`，不要把大文件和数据带进去

## 数据增强管线状态

- describe 阶段已完成：`probes/all_desc.jsonl`（17,937 段）
- explain 用 `--stage explain` + vLLM API（Qwen3.6-27B）
- facts 里 vk160-165 已需要替换为 LSHIFT/LCTRL/LALT 等名字（选项B 脚本）
