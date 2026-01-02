#!/usr/bin/env python3

import sys

import transformers


def patch_tunix_rollout_config_for_maxtext() -> None:
  # MaxText's `train_rl.py` may pass newer Tunix kwargs that are not yet released
  # in the latest `google-tunix` PyPI package. Patch Tunix dataclass configs to
  # ignore unknown kwargs for forward-compatibility.
  import inspect

  from tunix.rl.rollout import base_rollout
  from tunix.rl import rl_cluster as rl_cluster_lib

  def patch_dataclass_init_to_ignore_unknown_kwargs(dataclass_cls) -> None:
    if getattr(dataclass_cls, "_maxtext_plugin_patched", False):
      return

    supported = set(inspect.signature(dataclass_cls).parameters.keys())
    supported.discard("self")

    original_init = dataclass_cls.__init__

    def patched_init(self, *args, **kwargs):
      for key in list(kwargs.keys()):
        if key not in supported:
          kwargs.pop(key, None)
      original_init(self, *args, **kwargs)

    dataclass_cls.__init__ = patched_init  # type: ignore[method-assign]
    setattr(dataclass_cls, "_maxtext_plugin_patched", True)

  patch_dataclass_init_to_ignore_unknown_kwargs(base_rollout.RolloutConfig)
  patch_dataclass_init_to_ignore_unknown_kwargs(rl_cluster_lib.ClusterConfig)


def patch_hf_model_configs_for_qwen3_1p7b() -> None:
  # Tunix's vLLM adapter requires `config.model_name` to exist in MaxText's
  # `HF_MODEL_CONFIGS` mapping. MaxText doesn't currently allow `model_name=qwen3-1.7b`,
  # so we reuse the allowlisted `qwen3-0.6b` key but patch its HF config to match
  # Qwen3-1.7B.
  from MaxText.utils.ckpt_conversion.utils import hf_model_configs

  hf_model_configs.HF_MODEL_CONFIGS["qwen3-0.6b"] = transformers.Qwen3Config(
      vocab_size=151936,
      hidden_size=2048,
      intermediate_size=6144,
      num_hidden_layers=28,
      num_attention_heads=16,
      num_key_value_heads=8,
      head_dim=128,
      hidden_act="silu",
      max_position_embeddings=40960,
      rms_norm_eps=1.0e-6,
      rope_theta=1000000.0,
      tie_word_embeddings=True,
      torch_dtype="bfloat16",
  )


def main(argv: list[str]) -> int:
  patch_tunix_rollout_config_for_maxtext()
  patch_hf_model_configs_for_qwen3_1p7b()

  from MaxText.rl import train_rl

  train_rl.main(argv)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
