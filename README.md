# MaxText GRPO Plugin (Qwen3-1.7B · GSM8K · 10-step smoke test)

This repo is a **low-intrusion plugin** around upstream **MaxText** (vendored as a git submodule) to run a minimal
GRPO smoke test on **GSM8K** using **Qwen3-1.7B**.

Key goals:
- Keep upstream MaxText unmodified.
- Put all customizations under `plugin/`.
- Provide scripts to create a fresh TPU VM and run a 10-step GRPO job.

## What it runs
- Entry: `src.MaxText.rl.train_rl` (upstream MaxText)
- Config: `plugin/configs/rl_qwen3_1.7b_grpo_gsm8k_test.yml`
- Model config: `plugin/configs/models/qwen3-1.7b.yml`
- Steps: `num_batches=10` (with `num_iterations=1`, `num_epoch=1`)

## Quickstart (local control machine)
1) Ensure you have working `gcloud` + `gh` auth and a GCP project with TPU quota.
2) Create TPU + run remotely:
```bash
bash plugin/tpu/run_end_to_end.sh
```

See `plugin/DEV_PLAN.md` for the detailed plan and knobs.

