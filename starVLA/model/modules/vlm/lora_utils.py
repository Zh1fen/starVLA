from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    """Best-effort config getter for dict / OmegaConf / namespace-like objects."""
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)

    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass

    return getattr(container, key, default)


def get_qwenvl_config(config: Any) -> Any:
    framework_cfg = getattr(config, "framework", None)
    return _cfg_get(framework_cfg, "qwenvl", {}) or {}


def get_qwenvl_lora_config(config: Any) -> Any:
    return _cfg_get(get_qwenvl_config(config), "lora", {}) or {}


def is_lora_enabled(config: Any) -> bool:
    lora_cfg = get_qwenvl_lora_config(config)
    return bool(_cfg_get(lora_cfg, "enabled", False))


def get_qwen_default_lora_targets() -> list[str]:
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def resolve_lora_target_modules(target_modules: Any) -> list[str]:
    if target_modules is None or target_modules == "auto":
        return get_qwen_default_lora_targets()
    if isinstance(target_modules, str):
        return [item.strip() for item in target_modules.split(",") if item.strip()]
    if isinstance(target_modules, (list, tuple)):
        return [str(item).strip() for item in target_modules if str(item).strip()]
    raise TypeError(
        "LoRA `target_modules` must be `auto`, a comma-separated string, or a list/tuple of module names."
    )


def enable_gradient_checkpointing_if_needed(model: nn.Module, config: Any) -> bool:
    qwenvl_cfg = get_qwenvl_config(config)
    enabled = bool(_cfg_get(qwenvl_cfg, "enable_gradient_checkpointing", False))
    if not enabled:
        return False

    if not hasattr(model, "gradient_checkpointing_enable"):
        print(
            f"[LoRA] gradient checkpointing requested, but `{type(model).__name__}` does not expose "
            "`gradient_checkpointing_enable`.",
            flush=True,
        )
        return False

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print(f"[LoRA] gradient_checkpointing ENABLED on `{type(model).__name__}`", flush=True)
    return True


def apply_lora_if_enabled(model: nn.Module, config: Any) -> nn.Module:
    lora_cfg = get_qwenvl_lora_config(config)
    if not bool(_cfg_get(lora_cfg, "enabled", False)):
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise ImportError(
            "LoRA is enabled, but `peft` could not be imported. "
            "Install project dependencies again after adding `peft` to requirements."
        ) from exc

    target_modules = resolve_lora_target_modules(_cfg_get(lora_cfg, "target_modules", "auto"))
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(_cfg_get(lora_cfg, "r", 16)),
        lora_alpha=int(_cfg_get(lora_cfg, "alpha", 32)),
        lora_dropout=float(_cfg_get(lora_cfg, "dropout", 0.05)),
        bias=str(_cfg_get(lora_cfg, "bias", "none")),
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    print(
        f"[LoRA] enabled with r={peft_config.r}, alpha={peft_config.lora_alpha}, "
        f"dropout={peft_config.lora_dropout}, targets={target_modules}",
        flush=True,
    )
    return model


def get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        pass
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return torch.device("cpu")
