# Environments as Scaffold

论文 Environments as Scaffold: Enriching Feedback to Bootstrap Self-Evolving Agents in Long-Horizon Tasks 的相关代码，基于 [verl-agent](https://github.com/langfengQ/verl-agent)，包含 SciWorld 与 BFCL V3 Multi-Turn 两个环境。
 
## 代码结构

- `agent_system/environments/`：环境封装、反馈增强与数据划分。
- `agent_system/multi_turn_rollout/`：多轮交互与轨迹采样。
- `agent_system/reward_manager/`：回合奖励。
- `verl/`：模型训练与分布式基础设施。
- `scripts/`：安装、数据准备、训练和评估入口。

## 快速开始

训练需要已配置好的 Linux/CUDA 环境、PyTorch、vLLM 和 FlashAttention；SciWorld 还需要 Java。依赖安装入口见 `scripts/install.sh`，模型权重需自行准备。

```bash
export MODEL_PATH=/path/to/Qwen3-8B

# 标准环境：使用 GRPO 启动配置
bash scripts/run_bfcl.sh
bash scripts/run_sciworld.sh

# 在相同启动配置下启用反馈增强
ENV_NAME=bfcl_diversev3 bash scripts/run_bfcl.sh
ENV_NAME=sciworld_diversev7 bash scripts/run_sciworld.sh
```

两份启动脚本默认使用 8 张 GPU、16 个训练任务、每组 8 条轨迹，训练 300 个 epoch。BFCL/SciWorld 的验证任务数分别为 64/128，交互上限分别为 30/15 步。可通过 `SAVE_DIR` 指定输出目录，通过 `N_GPUS` 和 `TENSOR_PARALLEL_SIZE` 调整硬件配置；日志默认输出到控制台。

```bash
# 第一个参数指定推理引擎，后续参数覆盖训练配置
SAVE_DIR=./outputs/bfcl bash scripts/run_bfcl.sh vllm trainer.test_freq=5
```

## 通用训练与评估

通用入口与上述原始启动配置不同，默认训练 200 步，第 100 步后从 AG 切换为 OE，每 5 步评估一次。

```bash
# 格式：数据集 算法 反馈设置
bash scripts/train.sh sciworld grpo enriched
bash scripts/train.sh bfcl dapo enriched
bash scripts/train.sh sciworld gspo standard
bash scripts/train.sh sciworld grpo inconsistent

# MODEL_PATH 指向待评估的 Hugging Face 格式模型
bash scripts/evaluate.sh bfcl
bash scripts/evaluate.sh sciworld
```

反馈设置包括 `standard`（标准）、`enriched`（增强）和仅适用于 SciWorld 的 `inconsistent`（组内不一致）。数据准备由启动脚本自动完成，实际任务由环境加载。

## Related Projects

本项目建立在以下优秀开源工作的基础上，感谢相关作者和社区的贡献。

- [verl-agent](https://github.com/langfengQ/verl-agent)
- [SciWorld](https://github.com/allenai/ScienceWorld)
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [AppWorld] (https://github.com/StonyBrookNLP/appworld)
