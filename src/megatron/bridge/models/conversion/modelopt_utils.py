# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Iterator

import torch


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask

_NVFP4_AMAX_DENOMINATOR = 6.0 * 448.0
_QUANT_IGNORE_NAME_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
)
_EXPERT_NUMBER_PATTERNS = (
    re.compile(r"(local_experts\.)(\d+)(\.)"),
    re.compile(r"((?:weight|bias))(\d+)(?=$|\.)"),
    re.compile(r"(experts\.)(\d+)(\.)"),
)
_FUSED_MOE_NVFP4_NAME_MAP = {
    ".experts.gate_up_proj": ".experts.w13_weight",
    ".experts.down_proj": ".experts.w2_weight",
}


@dataclass(frozen=True)
class QuantMeta:
    """ModelOpt quantization metadata for one Megatron parameter."""

    qformat: str
    block_size: int
    weight_amax: torch.Tensor | None
    weight_scale_2: torch.Tensor | None = None


def _iter_quant_ignore_name_candidates(name: str) -> Iterator[str]:
    yield name
    for suffix in _QUANT_IGNORE_NAME_SUFFIXES:
        if name.endswith(suffix):
            yield name[: -len(suffix)]
            break

    alternate = name.removeprefix("model.") if name.startswith("model.") else f"model.{name}"

    yield alternate
    for suffix in _QUANT_IGNORE_NAME_SUFFIXES:
        if alternate.endswith(suffix):
            yield alternate[: -len(suffix)]
            break


def matches_quant_ignore_pattern(name: str, patterns: list[str]) -> bool:
    """Return whether a parameter name matches any ModelOpt ignore pattern."""
    return any(
        fnmatchcase(candidate, pattern)
        for candidate in _iter_quant_ignore_name_candidates(name)
        for pattern in patterns
    )


def is_modelopt_quantizable_weight_name(name: str) -> bool:
    """Return whether an exported HF tensor name should be ModelOpt-quantized."""
    return name.endswith(".weight") or any(name.endswith(suffix) for suffix in _FUSED_MOE_NVFP4_NAME_MAP)


def _is_same_tensor(param_weight: object, weight: object) -> bool:
    if param_weight is weight:
        return True
    if not isinstance(param_weight, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if param_weight.device.type == "meta" or weight.device.type == "meta":
        return False
    if (
        param_weight.device != weight.device
        or param_weight.dtype != weight.dtype
        or param_weight.layout != torch.strided
        or weight.layout != torch.strided
        or tuple(param_weight.shape) != tuple(weight.shape)
        or tuple(param_weight.stride()) != tuple(weight.stride())
    ):
        return False
    return (
        param_weight.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        and param_weight.storage_offset() == weight.storage_offset()
    )


def _iter_modelopt_weight_quantizers(
    module: torch.nn.Module,
) -> Iterator[tuple[object, object, bool]]:
    iter_weights = getattr(module, "iter_weights_for_calibration", None)
    if iter_weights is not None:
        for weight, weight_quantizer in iter_weights():
            if not _is_enabled_quantizer(weight_quantizer):
                continue
            yield (
                weight,
                weight_quantizer,
                _is_same_tensor(
                    getattr(module, "weight", None),
                    weight,
                ),
            )

    weight_quantizer = getattr(module, "weight_quantizer", None)
    if _is_enabled_quantizer(weight_quantizer):
        for weight_name, weight in module.named_parameters(recurse=False):
            if weight_name == "weight" or (weight_name.startswith("weight") and weight_name[6:].isdigit()):
                yield weight, weight_quantizer, weight_name == "weight"


def _is_enabled_quantizer(quantizer: object) -> bool:
    is_enabled = getattr(quantizer, "is_enabled", None)
    return bool(is_enabled)


def find_modelopt_weight_quantizer_and_module(
    module: torch.nn.Module,
    param_weight: object,
) -> tuple[object | None, torch.nn.Module | None]:
    """Find the enabled weight quantizer and owning module for ``param_weight``."""
    for _, candidate_module in module.named_modules():
        for weight, weight_quantizer, can_use_module in _iter_modelopt_weight_quantizers(candidate_module):
            if _is_same_tensor(param_weight, weight):
                if can_use_module:
                    return weight_quantizer, candidate_module
                quant_module = torch.nn.Module()
                quant_module.weight = weight
                quant_module.weight_quantizer = weight_quantizer
                input_quantizer = getattr(candidate_module, "input_quantizer", None)
                if input_quantizer is not None:
                    quant_module.input_quantizer = input_quantizer
                return weight_quantizer, quant_module

    return None, None


def _with_quant_meta_tensors(
    meta: QuantMeta,
    *,
    weight_amax: torch.Tensor | None,
    weight_scale_2: torch.Tensor | None,
) -> QuantMeta:
    return QuantMeta(
        qformat=meta.qformat,
        block_size=meta.block_size,
        weight_amax=weight_amax,
        weight_scale_2=weight_scale_2,
    )


def _clone_positive_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().float().abs().cpu()


def _slice_optional_quant_tensor(
    value: torch.Tensor | None,
    split: slice,
    leading_dim: int,
) -> torch.Tensor | None:
    if value is None or value.dim() == 0:
        return value
    if value.shape[0] != leading_dim:
        return value
    return value[split].contiguous()


def _slice_gated_quant_meta(meta: QuantMeta, hf_key: str) -> QuantMeta:
    """Slice fused ``[gate; up]`` metadata to match a split HF tensor."""
    if hf_key not in {"gate", "up"} or meta.weight_amax is None or meta.weight_amax.dim() == 0:
        return meta

    leading_dim = meta.weight_amax.shape[0]
    if leading_dim % 2 != 0:
        return meta

    midpoint = leading_dim // 2
    split = slice(0, midpoint) if hf_key == "gate" else slice(midpoint, leading_dim)
    return _with_quant_meta_tensors(
        meta,
        weight_amax=_slice_optional_quant_tensor(meta.weight_amax, split, leading_dim),
        weight_scale_2=meta.weight_scale_2,
    )


def _stack_optional_quant_tensors(
    values: list[torch.Tensor | None],
    *,
    hf_name: str,
    field_name: str,
) -> torch.Tensor | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise RuntimeError(f"Incomplete ModelOpt {field_name} metadata for grouped parameter {hf_name}")
    return torch.stack([value for value in values if value is not None], dim=0).contiguous()


def _stack_grouped_quant_meta(hf_name: str, expert_meta: dict[int, QuantMeta]) -> QuantMeta:
    if not expert_meta:
        raise RuntimeError(f"Missing ModelOpt metadata for grouped parameter {hf_name}")

    expected_experts = set(range(max(expert_meta) + 1))
    missing_experts = sorted(expected_experts.difference(expert_meta))
    if missing_experts:
        raise RuntimeError(f"Missing ModelOpt metadata for experts {missing_experts} of grouped parameter {hf_name}")

    metas = [expert_meta[idx] for idx in sorted(expert_meta)]
    qformat = metas[0].qformat
    block_size = metas[0].block_size
    for meta in metas[1:]:
        if meta.qformat != qformat or meta.block_size != block_size:
            raise RuntimeError(f"Inconsistent ModelOpt metadata for grouped parameter {hf_name}")

    return QuantMeta(
        qformat=qformat,
        block_size=block_size,
        weight_amax=_stack_optional_quant_tensors(
            [meta.weight_amax for meta in metas],
            hf_name=hf_name,
            field_name="weight_amax",
        ),
        weight_scale_2=_stack_optional_quant_tensors(
            [meta.weight_scale_2 for meta in metas],
            hf_name=hf_name,
            field_name="weight_scale_2",
        ),
    )


def _expert_param_template(param_name: str) -> str | None:
    for pattern in _EXPERT_NUMBER_PATTERNS:
        match = pattern.search(param_name)
        if match is None:
            continue
        return f"{param_name[: match.start(2)]}{{expert}}{param_name[match.end(2) :]}"
    return None


def _iter_grouped_quant_meta(
    task: WeightConversionTask,
    metadata: dict[str, QuantMeta],
) -> Iterator[tuple[int, QuantMeta]]:
    """Yield all synced per-expert metadata entries for a grouped export task."""
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    task_template = _expert_param_template(task.global_param_name)
    if task_template is None:
        raise ValueError(f"Expected expert parameter name for grouped export: {task.global_param_name}")

    for global_name, meta in metadata.items():
        if _expert_param_template(global_name) == task_template:
            yield extract_expert_number_from_param(global_name), meta


def build_hf_modelopt_quant_metadata(
    conversion_tasks: list[WeightConversionTask | None],
    metadata: dict[str, QuantMeta],
) -> dict[str, QuantMeta]:
    """Map Megatron ModelOpt metadata onto exported Hugging Face names."""
    hf_metadata: dict[str, QuantMeta] = {}
    grouped_metadata: dict[str, dict[int, QuantMeta]] = {}

    for task in conversion_tasks:
        if task is None or task.global_param_name not in metadata:
            continue

        meta = metadata[task.global_param_name]
        hf_param = task.mapping.hf_param
        if isinstance(hf_param, str):
            hf_items = (("", hf_param),)
        else:
            hf_items = tuple(hf_param.items())

        if getattr(task.mapping, "is_grouped_export", False):
            for expert_number, expert_meta in _iter_grouped_quant_meta(task, metadata):
                for hf_key, hf_name in hf_items:
                    grouped_metadata.setdefault(hf_name, {})[expert_number] = _slice_gated_quant_meta(
                        expert_meta,
                        hf_key,
                    )
            continue

        for _, hf_name in hf_items:
            hf_metadata[hf_name] = meta

    for hf_name, expert_meta in grouped_metadata.items():
        hf_metadata[hf_name] = _stack_grouped_quant_meta(hf_name, expert_meta)

    return hf_metadata


def collect_modelopt_quant_metadata(
    conversion_tasks: list[WeightConversionTask | None],
) -> dict[str, QuantMeta]:
    """Collect ModelOpt quantization metadata from conversion task modules."""
    from modelopt.torch.export.quant_utils import (
        QUANTIZATION_NONE,
        QUANTIZATION_NVFP4,
        QUANTIZATION_W4A16_NVFP4,
        get_quantization_format,
        get_weight_block_size,
    )
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    metadata: dict[str, QuantMeta] = {}
    for task in conversion_tasks:
        if task is None or task.megatron_module is None or task.param_weight is None:
            continue

        weight_quantizer, quant_module = find_modelopt_weight_quantizer_and_module(
            task.megatron_module,
            task.param_weight,
        )
        if weight_quantizer is None:
            continue

        qformat = get_quantization_format(quant_module)
        if qformat == QUANTIZATION_NONE:
            continue

        block_size = get_weight_block_size(quant_module)
        if block_size == 0:
            continue
        weight_amax = getattr(weight_quantizer, "_amax", None)
        weight_scale_2 = None
        if qformat in (QUANTIZATION_NVFP4, QUANTIZATION_W4A16_NVFP4):
            weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(weight_quantizer)

        metadata[task.global_param_name] = QuantMeta(
            qformat=qformat,
            block_size=block_size,
            weight_amax=_clone_positive_cpu(weight_amax),
            weight_scale_2=_clone_positive_cpu(weight_scale_2),
        )
    return metadata


def sync_modelopt_quant_metadata(metadata: dict[str, QuantMeta], group=None) -> None:
    """Synchronize ModelOpt quantization metadata across a distributed group."""
    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, QuantMeta] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, metadata, group=group)

    for rank_metadata in gathered:
        if rank_metadata:
            metadata.update(rank_metadata)


def _reshape_nvfp4_weight_scale_2_for_compute(
    weight: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    if weight.dim() != 3 or weight_scale_2.dim() == 0:
        return weight_scale_2
    if weight_scale_2.dim() == 1:
        return weight_scale_2.view(-1, 1, 1)
    if weight_scale_2.dim() == 2:
        if weight_scale_2.shape[1] == 1:
            return weight_scale_2.view(weight_scale_2.shape[0], 1, 1)
        if weight.shape[1] % weight_scale_2.shape[1] == 0:
            repeat = weight.shape[1] // weight_scale_2.shape[1]
            return weight_scale_2.repeat_interleave(repeat, dim=1).unsqueeze(-1)
    return weight_scale_2


def compute_nvfp4_weight_scale(
    weight: torch.Tensor,
    block_size: int,
    weight_amax: torch.Tensor | None = None,
    weight_scale_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the NVFP4 per-block weight scale tensor for ModelOpt export."""
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    if weight_amax is not None:
        scale_2_for_export = weight_amax.to(weight.device).float().abs() / _NVFP4_AMAX_DENOMINATOR
    elif weight_scale_2 is not None:
        scale_2_for_export = weight_scale_2.to(weight.device).float().abs()
    else:
        raise RuntimeError("Missing ModelOpt weight amax for NVFP4 export")

    scale_2_for_compute = _reshape_nvfp4_weight_scale_2_for_compute(
        weight,
        scale_2_for_export,
    )

    weight_scale = NVFP4QTensor.get_weights_scaling_factor(
        weight,
        block_size,
        weights_scaling_factor_2=scale_2_for_compute,
        keep_high_precision=True,
    )[0]
    weight_scale = weight_scale.to(torch.float32).abs()
    weight_scale[weight_scale == 0] = 1.0
    weight_scale = weight_scale.to(torch.float8_e4m3fn)
    return weight_scale, scale_2_for_export


def _nvfp4_export_names(name: str) -> tuple[str, str, str]:
    if name.endswith(".weight"):
        base = name[: -len(".weight")]
        return name, f"{base}.weight_scale", f"{base}.weight_scale_2"
    for hf_suffix, vllm_suffix in _FUSED_MOE_NVFP4_NAME_MAP.items():
        if name.endswith(hf_suffix):
            base = name[: -len(hf_suffix)] + vllm_suffix
            return base, f"{base}_scale", f"{base}_scale_2"
    raise ValueError(f"Expected quantizable NVFP4 export parameter name: {name}")


def _format_nvfp4_weight_scale_2_for_export(
    weight_name: str,
    weight: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    if weight_name.endswith("w13_weight"):
        if weight_scale_2.dim() == 0:
            weight_scale_2 = weight_scale_2.expand(weight.shape[0])
        if weight_scale_2.dim() == 1:
            return weight_scale_2[:, None].expand(-1, 2).contiguous()
        if weight_scale_2.dim() == 2 and weight_scale_2.shape[1] == 1:
            return weight_scale_2.expand(-1, 2).contiguous()
    if weight_name.endswith("w2_weight"):
        if weight_scale_2.dim() == 0:
            return weight_scale_2.expand(weight.shape[0]).contiguous()
        if weight_scale_2.dim() == 2 and weight_scale_2.shape[1] == 1:
            return weight_scale_2[:, 0].contiguous()
    return weight_scale_2


def quantize_nvfp4_weight(
    name: str,
    weight: torch.Tensor,
    meta: QuantMeta,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield NVFP4 quantized weight tensors and associated scale tensors."""
    from modelopt.torch.export.quant_utils import to_quantized_weight

    weight_name, weight_scale_name, weight_scale_2_name = _nvfp4_export_names(name)
    weight_scale, weight_scale_2 = compute_nvfp4_weight_scale(
        weight,
        meta.block_size,
        weight_amax=meta.weight_amax,
        weight_scale_2=meta.weight_scale_2,
    )
    if not torch.isfinite(weight_scale_2).all() or not torch.all(weight_scale_2 > 0):
        raise RuntimeError(f"Invalid ModelOpt weight_scale_2 for quantized parameter {name}: {weight_scale_2}")
    weight_scale_2_for_quant = weight_scale_2
    if weight.dim() == 3:
        weight_scale_2_for_quant = _reshape_nvfp4_weight_scale_2_for_compute(
            weight,
            weight_scale_2,
        )
    if weight.dim() == 2 and weight_scale_2.numel() == 1:
        weight_scale_2_for_quant = weight_scale_2.reshape(())

    quantized = to_quantized_weight(
        weight,
        weight_scale,
        meta.qformat,
        weight_scale_2_for_quant,
        meta.block_size,
    )

    yield weight_name, quantized.detach()
    yield weight_scale_name, weight_scale.detach()
    yield (
        weight_scale_2_name,
        _format_nvfp4_weight_scale_2_for_export(
            weight_name,
            weight,
            weight_scale_2,
        ).detach(),
    )


def get_modelopt_quant_exporter(quant_mode: str):
    """Return the ModelOpt quantization format and exporter for a quantization mode."""
    from modelopt.torch.export.quant_utils import (
        QUANTIZATION_NVFP4,
        QUANTIZATION_W4A16_NVFP4,
    )

    mode_to_qformat = {
        "nvfp4": QUANTIZATION_NVFP4,
        "w4a16_nvfp4": QUANTIZATION_W4A16_NVFP4,
    }
    qformat = mode_to_qformat.get(quant_mode.lower())
    if qformat is None:
        raise ValueError(f"Unsupported ModelOpt quant_mode: {quant_mode}")
    return qformat, quantize_nvfp4_weight
