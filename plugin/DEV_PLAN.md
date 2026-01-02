# 开发/执行计划（低侵入 MaxText 插件：Qwen3-1.7B · GSM8K · GRPO · 10 steps）

目标：不修改上游 `maxtext` 源码，仅在 `plugin/` 增加配置与脚本，在 **TPU v6e-8（europe-west4-a）** 上完成 **GSM8K 的 GRPO 训练 smoke test（总步数=10）**，并用 `gh` 推送到 GitHub，使用 `gcloud` 新建 TPU 并让 TPU 从 GitHub 拉取运行。

---

## 0. 约定与前置

- 运行入口使用上游：`python3 -m src.MaxText.rl.train_rl ...`
- **不**向 `third_party/maxtext`（上游 MaxText 子模块）打补丁；所有新增放在 `plugin/`。
- 训练是 smoke test：只跑 10 个 step（通过 `num_batches=10` 达到）。

需要你本地（控制机）具备：
- `gcloud` 已登录且项目可用（已启用 `compute.googleapis.com`、`tpu.googleapis.com`）
- `gh` 已登录并有创建/推送仓库权限

---

## 1. 代码结构（本仓库）

- `third_party/maxtext`：上游 MaxText（git submodule）
- `plugin/configs/rl_qwen3_1.7b_grpo_gsm8k_test.yml`：RL 配置（继承上游 `rl.yml`，覆盖为 10-step）
- `plugin/configs/models/qwen3-1.7b.yml`：Qwen3-1.7B 模型配置（从 HF `config.json` 抽取核心超参）
- `plugin/scripts/run_grpo_gsm8k_test.sh`：在 TPU VM 上执行训练
- `plugin/tpu/*`：TPU 创建、远程 bootstrap 与一键 end-to-end

---

## 2. 配置策略（10 steps）

上游 `train_rl.py` 的训练步数是：

`max_train_steps = num_batches * num_iterations * train_fraction * num_epoch`

因此我们设置：
- `num_batches: 10`
- `num_iterations: 1`
- `train_fraction: 1.0`
- `num_epoch: 1`

---

## 3. GitHub 推送

1) 初始化 git（本仓库）
2) 添加 `third_party/maxtext` 子模块（指向上游 MaxText）
3) 提交并用 `gh` 创建仓库、push

---

## 4. TPU 执行（新建 v6e-8，避免使用现有 TPU）

1) 使用 `gcloud compute tpus tpu-vm create` 创建新 TPU VM：
   - zone：`europe-west4-a`
   - accelerator：`v6e-8`
   - 如遇到 `Insufficient capacity`，可用 `--spot`（脚本默认自动 fallback，或显式设置 `TPU_VM_CREATE_FLAGS=--spot`）
2) 通过 `gcloud compute tpus tpu-vm ssh --command ...` 在 TPU VM 上：
   - `git clone --recursive` 本仓库
   - 创建 venv 并安装 MaxText 依赖（含 post-training：tunix + vllm-tpu）
   - 运行 `plugin/scripts/run_grpo_gsm8k_test.sh`
   - **长任务用 nohup 后台跑**：日志写入 `~/qwen3_grpo_<run_name>.log`，用 `tail -n 100` 轮询查看进度
3) 训练完成后：
   - 保存输出在 TPU VM 本地目录（默认 `~/runs/<run_name>`）
   -（可选）用 `gcloud compute tpus tpu-vm stop` 停机省成本

---

## 5. 验收标准（最小可用）

- TPU VM 为**新建**，不是现有机器
- GRPO 训练启动成功并跑完 10 steps（日志能看到 step 前进，且 `RL Training Completed Successfully!`）
- 输出目录包含 checkpoints/tensorboard/metrics（任意一种即可证明写出）
