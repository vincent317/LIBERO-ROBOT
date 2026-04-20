#!/usr/bin/env python3
"""Local helpers for using LeRobot ACT without importing the broken policies package."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch


LOCAL_CACHE_ROOT = Path("/tmp/libero_hf_cache")
LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_ROOT / "hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", str(LOCAL_CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_LEROBOT_HOME", str(LOCAL_CACHE_ROOT / "lerobot"))
os.environ.setdefault("TORCH_HOME", str(LOCAL_CACHE_ROOT / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))


LEROBOT_SITE_PACKAGES = Path(
    "/home/ec2-user/miniconda3/envs/lerobot/lib/python3.12/site-packages/lerobot"
)


def _ensure_stub_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_lerobot_act() -> None:
    """Load only the ACT-related LeRobot modules and bypass lerobot.policies.__init__."""
    importlib.import_module("lerobot")

    _ensure_stub_package("lerobot.policies", LEROBOT_SITE_PACKAGES / "policies")
    _ensure_stub_package("lerobot.policies.act", LEROBOT_SITE_PACKAGES / "policies" / "act")

    _load_module(
        "lerobot.policies.utils",
        LEROBOT_SITE_PACKAGES / "policies" / "utils.py",
    )
    _load_module(
        "lerobot.policies.pretrained",
        LEROBOT_SITE_PACKAGES / "policies" / "pretrained.py",
    )
    _load_module(
        "lerobot.policies.act.configuration_act",
        LEROBOT_SITE_PACKAGES / "policies" / "act" / "configuration_act.py",
    )
    _load_module(
        "lerobot.policies.act.modeling_act",
        LEROBOT_SITE_PACKAGES / "policies" / "act" / "modeling_act.py",
    )


def get_act_classes():
    bootstrap_lerobot_act()
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    return ACTConfig, ACTPolicy


def get_policy_classes(use_task_id_conditioning: bool = False):
    if use_task_id_conditioning:
        from local_multitask_act import ACTTaskIDConfig, ACTTaskIDPolicy

        return ACTTaskIDConfig, ACTTaskIDPolicy
    return get_act_classes()


def get_policy_classes_for_checkpoint(checkpoint: str | Path):
    checkpoint_path = Path(checkpoint)
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy_type = config.get("type")

    if policy_type == "act_task_id_shared_encoder":
        from local_multitask_act_shared_task_encoder import (
            ACTTaskIDSharedEncoderConfig,
            ACTTaskIDSharedEncoderPolicy,
        )

        return ACTTaskIDSharedEncoderConfig, ACTTaskIDSharedEncoderPolicy
    if policy_type == "act_task_id_vae_encoder":
        from local_multitask_act_vae_encoder import ACTTaskIDVAEEncoderConfig, ACTTaskIDVAEEncoderPolicy

        return ACTTaskIDVAEEncoderConfig, ACTTaskIDVAEEncoderPolicy
    if policy_type == "act_task_id":
        from local_multitask_act import ACTTaskIDConfig, ACTTaskIDPolicy

        return ACTTaskIDConfig, ACTTaskIDPolicy
    return get_act_classes()


def get_policy_feature_utils():
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )

    return dataset_to_policy_features, NormalizerProcessorStep, UnnormalizerProcessorStep


def device_from_arg(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def serialize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: serialize_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize_jsonable(data), indent=2), encoding="utf-8")
