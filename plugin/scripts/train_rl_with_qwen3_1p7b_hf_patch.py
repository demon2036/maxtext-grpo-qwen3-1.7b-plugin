#!/usr/bin/env python3

import os
import sys
import types

import transformers


_TUNIX_CLUSTER_CONFIG_ROLE_AXIS_RULES_BY_ID: dict[int, dict] = {}


def patch_tunix_rollout_config_for_maxtext() -> None:
  # MaxText's `train_rl.py` may pass newer Tunix kwargs that are not yet released
  # in the latest `google-tunix` PyPI package. Patch Tunix dataclass configs to
  # ignore unknown kwargs for forward-compatibility.
  import inspect

  import gc
  import operator

  import jax
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from tunix.rl.rollout import base_rollout
  from tunix.rl import rl_cluster as rl_cluster_lib
  from tunix.rl import utils as rl_utils
  from tunix.rl import reshard

  desired_rollout_engine = os.environ.get("MAXTEXT_PLUGIN_ROLLOUT_ENGINE", "").strip()
  if not desired_rollout_engine:
    desired_rollout_engine = "vanilla"

  def patch_dataclass_init_to_ignore_unknown_kwargs(
      dataclass_cls, *, keep_kwargs=None, override_kwargs=None
  ) -> None:
    keep_kwargs = set(keep_kwargs or [])
    override_kwargs = dict(override_kwargs or {})
    if getattr(dataclass_cls, "_maxtext_plugin_patched", False):
      return

    supported = set(inspect.signature(dataclass_cls).parameters.keys())
    supported.discard("self")

    original_init = dataclass_cls.__init__
    dataclass_params = getattr(dataclass_cls, "__dataclass_params__", None)
    is_frozen = bool(getattr(dataclass_params, "frozen", False))

    def patched_init(self, *args, **kwargs):
      preserved = {k: kwargs.pop(k) for k in keep_kwargs if k in kwargs}
      for key in list(kwargs.keys()):
        if key not in supported:
          kwargs.pop(key, None)
      kwargs.update(override_kwargs)
      original_init(self, *args, **kwargs)
      if preserved:
        if is_frozen:
          # ClusterConfig is frozen; stash extra data externally keyed by object id.
          _TUNIX_CLUSTER_CONFIG_ROLE_AXIS_RULES_BY_ID[id(self)] = preserved
        else:
          for k, v in preserved.items():
            setattr(self, k, v)

    dataclass_cls.__init__ = patched_init  # type: ignore[method-assign]
    setattr(dataclass_cls, "_maxtext_plugin_patched", True)

  patch_dataclass_init_to_ignore_unknown_kwargs(base_rollout.RolloutConfig)
  patch_dataclass_init_to_ignore_unknown_kwargs(
      rl_cluster_lib.ClusterConfig,
      keep_kwargs={"role_to_logical_axis_rule"},
      override_kwargs={"rollout_engine": desired_rollout_engine},
  )

  rollout_cluster_cls = rl_cluster_lib.RLCluster
  if not getattr(rollout_cluster_cls, "_maxtext_plugin_patched", False):
    original_load_model = rollout_cluster_cls._load_model

    def meshes_equivalent(a, b) -> bool:
      if a is b:
        return True
      if getattr(a, "empty", False) and getattr(b, "empty", False):
        return True
      try:
        if a.axis_names != b.axis_names:
          return False
        if a.shape != b.shape:
          return False
        return list(a.devices.flat) == list(b.devices.flat)
      except Exception:  # pylint: disable=broad-except
        return False

    def patched_load_model(self, model_or_path, mesh, data_type=None):  # pylint: disable=missing-docstring
      if not isinstance(model_or_path, nnx.Module):
        return original_load_model(self, model_or_path, mesh, data_type)

      model_mesh = rl_utils.get_pytree_mesh_info(nnx.state(model_or_path))
      original_shardings = jax.tree_util.tree_map(
          lambda x: x.sharding, nnx.state(model_or_path)
      )
      is_on_device = jax.tree_util.tree_reduce(
          operator.or_,
          jax.tree.map(
              lambda x: x.memory_kind == self._default_memory_kind,
              original_shardings,
          ),
      )

      if not mesh.empty and not meshes_equivalent(model_mesh, mesh):
        role_to_axis_rules = _TUNIX_CLUSTER_CONFIG_ROLE_AXIS_RULES_BY_ID.get(id(self.cluster_config), {}).get(
            "role_to_logical_axis_rule", {}
        )
        role_for_mesh = None
        for role, role_mesh in getattr(self, "r2m", {}).items():
          if role_mesh is mesh:
            role_for_mesh = role
            break
        axis_rules = (
            role_to_axis_rules.get(role_for_mesh)
            or role_to_axis_rules.get(rl_cluster_lib.Role.ACTOR)
            or role_to_axis_rules.get(rl_cluster_lib.Role.REFERENCE)
        )

        graph, state = nnx.split(model_or_path)
        if axis_rules:
          with nn_partitioning.axis_rules(axis_rules):
            dst_shardings = jax.tree_util.tree_map(
                lambda x: jax.sharding.NamedSharding(
                    mesh,
                    x,
                    memory_kind=self._default_memory_kind
                    if is_on_device
                    else "pinned_host",
                ),
                nnx.get_partition_spec(state),
            )
        else:
          dst_shardings = jax.tree_util.tree_map(
              lambda x: jax.sharding.NamedSharding(
                  mesh,
                  x,
                  memory_kind=self._default_memory_kind
                  if is_on_device
                  else "pinned_host",
              ),
              nnx.get_partition_spec(state),
          )
        if data_type and data_type != jax.tree.leaves(state)[0].dtype:
          tmp_state = jax.tree.map(lambda x: x.astype(data_type), state)
        else:
          tmp_state = state
        model_or_path = nnx.merge(
            graph, reshard.reshard_pytree(tmp_state, dst_shardings)
        )
        del tmp_state
        gc.collect()

      if is_on_device and self.cluster_config.offload_to_cpu:
        graph, state = nnx.split(model_or_path)
        new_params = rl_utils.put_params_on_memory_kind(state, "pinned_host")
        model_or_path = nnx.merge(graph, new_params)
      return model_or_path

    rollout_cluster_cls._load_model = patched_load_model  # type: ignore[method-assign]
    setattr(rollout_cluster_cls, "_maxtext_plugin_patched", True)


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


def patch_maxtext_train_rl_for_vanilla_rollout() -> None:
  from MaxText.rl import train_rl

  original_get_maxtext_model = train_rl.get_maxtext_model
  if getattr(original_get_maxtext_model, "_maxtext_plugin_patched", False):
    return

  def patched_get_maxtext_model(config, devices=None):
    tunix_model, mesh = original_get_maxtext_model(config, devices=devices)
    tunix_model.config = types.SimpleNamespace(
        num_layers=int(config.num_decoder_layers),
        num_kv_heads=int(config.num_kv_heads),
        head_dim=int(config.head_dim),
    )
    return tunix_model, mesh

  patched_get_maxtext_model._maxtext_plugin_patched = True  # type: ignore[attr-defined]
  train_rl.get_maxtext_model = patched_get_maxtext_model


def main(argv: list[str]) -> int:
  patch_tunix_rollout_config_for_maxtext()
  patch_hf_model_configs_for_qwen3_1p7b()

  from MaxText.rl import train_rl

  rollout_engine = os.environ.get("MAXTEXT_PLUGIN_ROLLOUT_ENGINE", "").strip() or "vanilla"
  if rollout_engine == "vanilla":
    patch_maxtext_train_rl_for_vanilla_rollout()

  train_rl.main(argv)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
