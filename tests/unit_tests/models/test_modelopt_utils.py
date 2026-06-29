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

from types import SimpleNamespace

import pytest
import torch


quant_utils = pytest.importorskip("modelopt.torch.export.quant_utils")
QUANTIZATION_NONE = quant_utils.QUANTIZATION_NONE
QUANTIZATION_NVFP4 = quant_utils.QUANTIZATION_NVFP4

from megatron.bridge.models.conversion import modelopt_utils
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.modelopt_utils import (
    QuantMeta,
    build_hf_modelopt_quant_metadata,
    collect_modelopt_quant_metadata,
    find_modelopt_weight_quantizer_and_module,
    get_modelopt_quant_exporter,
    is_modelopt_quantizable_weight_name,
    matches_quant_ignore_pattern,
    quantize_nvfp4_weight,
    sync_modelopt_quant_metadata,
)


def _task(
    global_param_name,
    hf_param,
    *,
    megatron_module=None,
    param_weight=None,
):
    return SimpleNamespace(
        global_param_name=global_param_name,
        mapping=SimpleNamespace(hf_param=hf_param),
        megatron_module=megatron_module,
        param_weight=param_weight,
    )


def _quant_meta(qformat=QUANTIZATION_NVFP4):
    return QuantMeta(
        qformat=qformat,
        block_size=16,
        weight_amax=torch.tensor([1.0]),
        weight_scale_2=torch.tensor([1.0 / (6.0 * 448.0)]),
    )


def _bridge_for_export(conversion_tasks, exported_weights):
    class FakeBridge:
        hf_pretrained = object()
        _model_bridge = SimpleNamespace(
            build_conversion_tasks=lambda *_args, **_kwargs: conversion_tasks,
        )

        def __init__(self):
            self.export_calls = []

        def export_hf_weights(self, model, **kwargs):
            self.export_calls.append((model, kwargs))
            yield from exported_weights

    return FakeBridge()


def test_matches_quant_ignore_pattern_handles_model_prefix_and_scale_suffixes():
    ignore_patterns = [
        "lm_head",
        "*self_attn*",
        "*mlp.gate",
        "*router*",
    ]

    assert matches_quant_ignore_pattern(
        "model.layers.0.self_attn.o_proj.weight",
        ignore_patterns,
    )
    assert matches_quant_ignore_pattern(
        "layers.0.self_attn.o_proj.weight",
        ignore_patterns,
    )
    assert matches_quant_ignore_pattern("model.layers.0.mlp.gate.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("model.layers.0.router.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("lm_head.weight", ignore_patterns)
    assert matches_quant_ignore_pattern("model.layers.0.mlp.gate.weight_scale", ignore_patterns)
    assert not matches_quant_ignore_pattern(
        "model.layers.0.mlp.experts.0.w1.weight",
        ignore_patterns,
    )


def test_find_modelopt_weight_quantizer_uses_proxy_for_custom_weight():
    class FakeQuantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.custom_weight = torch.nn.Parameter(torch.ones(1))
            self.weight_quantizer = SimpleNamespace(is_enabled=True)
            self.custom_quantizer = SimpleNamespace(is_enabled=True)
            self.input_quantizer = SimpleNamespace(is_enabled=True)

        def iter_weights_for_calibration(self):
            yield self.custom_weight, self.custom_quantizer

    module = FakeQuantModule()

    weight_quantizer, quant_module = find_modelopt_weight_quantizer_and_module(
        module,
        module.custom_weight,
    )

    assert weight_quantizer is module.custom_quantizer
    assert quant_module is not module
    assert quant_module.weight is module.custom_weight
    assert quant_module.weight_quantizer is module.custom_quantizer
    assert quant_module.input_quantizer is module.input_quantizer


def test_find_modelopt_weight_quantizer_returns_owner_for_weight_param():
    class FakeQuantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.weight_quantizer = SimpleNamespace(is_enabled=True)

        def iter_weights_for_calibration(self):
            yield self.weight, self.weight_quantizer

    module = FakeQuantModule()

    weight_quantizer, quant_module = find_modelopt_weight_quantizer_and_module(
        module,
        module.weight,
    )

    assert weight_quantizer is module.weight_quantizer
    assert quant_module is module


def test_collect_modelopt_quant_metadata_skips_unquantized_tasks(monkeypatch):
    quantizer_amax = torch.tensor([-2688.0])
    quant_module = SimpleNamespace(weight_quantizer=SimpleNamespace(_amax=quantizer_amax))
    unquantized_module = SimpleNamespace(weight_quantizer=SimpleNamespace(_amax=torch.tensor([1.0])))
    blockless_module = SimpleNamespace(weight_quantizer=SimpleNamespace(_amax=torch.tensor([2.0])))

    qformat_by_module = {
        id(quant_module): QUANTIZATION_NVFP4,
        id(unquantized_module): QUANTIZATION_NONE,
        id(blockless_module): QUANTIZATION_NVFP4,
    }
    block_size_by_module = {
        id(quant_module): 16,
        id(unquantized_module): 16,
        id(blockless_module): 0,
    }

    monkeypatch.setattr(
        quant_utils,
        "get_quantization_format",
        lambda module: qformat_by_module[id(module)],
    )
    monkeypatch.setattr(
        quant_utils,
        "get_weight_block_size",
        lambda module: block_size_by_module[id(module)],
    )
    monkeypatch.setattr(
        modelopt_utils,
        "find_modelopt_weight_quantizer_and_module",
        lambda module, _weight: (module.weight_quantizer, module),
    )

    metadata = collect_modelopt_quant_metadata(
        [
            None,
            _task(
                "missing.module.weight",
                "hf.missing.weight",
                megatron_module=None,
                param_weight=torch.empty(1),
            ),
            _task(
                "missing.param.weight",
                "hf.missing_param.weight",
                megatron_module=quant_module,
            ),
            _task(
                "unquantized.weight",
                "hf.unquantized.weight",
                megatron_module=unquantized_module,
                param_weight=torch.empty(1),
            ),
            _task(
                "blockless.weight",
                "hf.blockless.weight",
                megatron_module=blockless_module,
                param_weight=torch.empty(1),
            ),
            _task(
                "quantized.weight",
                "hf.quantized.weight",
                megatron_module=quant_module,
                param_weight=torch.empty(1),
            ),
        ]
    )

    assert list(metadata) == ["quantized.weight"]
    assert metadata["quantized.weight"].qformat == QUANTIZATION_NVFP4
    assert metadata["quantized.weight"].block_size == 16
    torch.testing.assert_close(
        metadata["quantized.weight"].weight_amax,
        quantizer_amax.abs(),
    )
    torch.testing.assert_close(
        metadata["quantized.weight"].weight_scale_2,
        torch.tensor([1.0]),
    )
    assert metadata["quantized.weight"].weight_amax.data_ptr() != quantizer_amax.data_ptr()


def test_sync_modelopt_quant_metadata_merges_gathered_rank_metadata(monkeypatch):
    rank1_meta = QuantMeta(
        qformat=QUANTIZATION_NVFP4,
        block_size=16,
        weight_amax=torch.tensor([2.0]),
    )
    metadata = {
        "rank0.weight": QuantMeta(
            qformat=QUANTIZATION_NVFP4,
            block_size=16,
            weight_amax=torch.tensor([1.0]),
        )
    }

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def fake_all_gather_object(gathered, local_metadata, group=None):
        gathered[0] = dict(local_metadata)
        gathered[1] = {"rank1.weight": rank1_meta}

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    sync_modelopt_quant_metadata(metadata, group=object())

    assert set(metadata) == {"rank0.weight", "rank1.weight"}
    assert metadata["rank1.weight"] is rank1_meta


def test_build_hf_modelopt_quant_metadata_stacks_synced_grouped_experts():
    hf_name = "model.layers.0.mlp.experts.gate_up_proj.weight"
    task = SimpleNamespace(
        global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight0",
        mapping=SimpleNamespace(hf_param=hf_name, is_grouped_export=True),
    )
    metadata = {
        f"decoder.layers.0.mlp.experts.linear_fc1.weight{expert_idx}": QuantMeta(
            qformat=QUANTIZATION_NVFP4,
            block_size=16,
            weight_amax=torch.tensor([float(expert_idx)]),
            weight_scale_2=torch.tensor([float(expert_idx + 10)]),
        )
        for expert_idx in range(4)
    }

    hf_metadata = build_hf_modelopt_quant_metadata([task], metadata)

    meta = hf_metadata[hf_name]
    assert meta.qformat == QUANTIZATION_NVFP4
    assert meta.block_size == 16
    torch.testing.assert_close(
        meta.weight_amax,
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
    )
    torch.testing.assert_close(
        meta.weight_scale_2,
        torch.tensor([[10.0], [11.0], [12.0], [13.0]]),
    )


def test_quantize_nvfp4_weight_uses_modelopt_scale_export_and_emits_scale_names(monkeypatch):
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured.update(
            weight=weight,
            weight_scale=weight_scale,
            qformat=qformat,
            weight_scale_2=weight_scale_2,
            block_size=block_size,
        )
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    tensors = dict(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.up_proj.weight",
            torch.tensor([[-1.0, 0.25, 0.5, 2.0]], dtype=torch.float32),
            QuantMeta(
                qformat=QUANTIZATION_NVFP4,
                block_size=4,
                weight_amax=torch.tensor([2688.0]),
                weight_scale_2=torch.tensor([1.0]),
            ),
        )
    )

    assert set(tensors) == {
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight_scale",
        "model.layers.0.mlp.up_proj.weight_scale_2",
    }
    assert tensors["model.layers.0.mlp.up_proj.weight"].dtype == torch.uint8
    assert tensors["model.layers.0.mlp.up_proj.weight_scale"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.up_proj.weight_scale_2"],
        torch.tensor([1.0]),
    )
    assert captured["qformat"] == QUANTIZATION_NVFP4
    assert captured["block_size"] == 4
    assert captured["weight_scale"].dtype == torch.float8_e4m3fn
    assert (captured["weight_scale"].to(torch.float32) >= 0).all()
    assert captured["weight_scale_2"].dim() == 0
    torch.testing.assert_close(captured["weight_scale_2"], torch.tensor(1.0))


def test_quantize_nvfp4_weight_exports_fused_moe_internal_names(monkeypatch):
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured.update(
            weight=weight,
            weight_scale=weight_scale,
            qformat=qformat,
            weight_scale_2=weight_scale_2,
            block_size=block_size,
        )
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    tensors = dict(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.experts.gate_up_proj",
            torch.ones(2, 4, 4, dtype=torch.float32),
            QuantMeta(
                qformat=QUANTIZATION_NVFP4,
                block_size=4,
                weight_amax=torch.tensor([[2688.0], [1344.0]]),
                weight_scale_2=torch.tensor([[1.0], [0.5]]),
            ),
        )
    )

    assert set(tensors) == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w13_weight_scale",
        "model.layers.0.mlp.experts.w13_weight_scale_2",
    }
    assert tensors["model.layers.0.mlp.experts.w13_weight"].dtype == torch.uint8
    assert tensors["model.layers.0.mlp.experts.w13_weight_scale"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        tensors["model.layers.0.mlp.experts.w13_weight_scale_2"],
        torch.tensor([[1.0, 1.0], [0.5, 0.5]]),
    )
    assert captured["weight_scale_2"].shape == (2, 1, 1)


def test_quantize_nvfp4_weight_recomputes_global_scale_from_amax(monkeypatch):
    captured = {}

    def fake_to_quantized_weight(
        weight,
        weight_scale,
        qformat,
        weight_scale_2,
        block_size,
    ):
        captured["weight_scale_2"] = weight_scale_2.detach().cpu()
        return torch.zeros(weight.shape, dtype=torch.uint8, device=weight.device)

    monkeypatch.setattr(quant_utils, "to_quantized_weight", fake_to_quantized_weight)

    list(
        quantize_nvfp4_weight(
            "model.layers.0.mlp.down_proj.weight",
            torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=torch.float32),
            QuantMeta(
                qformat=QUANTIZATION_NVFP4,
                block_size=4,
                weight_amax=torch.tensor([2688.0]),
                weight_scale_2=torch.tensor([123.0]),
            ),
        )
    )

    assert captured["weight_scale_2"].dim() == 0
    torch.testing.assert_close(captured["weight_scale_2"], torch.tensor(1.0))


def test_quantize_nvfp4_weight_requires_quantizable_name():
    with pytest.raises(ValueError, match="Expected quantizable NVFP4 export parameter name"):
        list(
            quantize_nvfp4_weight(
                "model.layers.0.mlp.up_proj",
                torch.ones(1, 4),
                QuantMeta(
                    qformat=QUANTIZATION_NVFP4,
                    block_size=4,
                    weight_amax=torch.tensor([1.0]),
                    weight_scale_2=torch.tensor([1.0]),
                ),
            )
        )


def test_is_modelopt_quantizable_weight_name_includes_fused_moe_base_names():
    assert is_modelopt_quantizable_weight_name("model.layers.0.mlp.down_proj.weight")
    assert is_modelopt_quantizable_weight_name("model.layers.0.mlp.experts.gate_up_proj")
    assert is_modelopt_quantizable_weight_name("model.layers.0.mlp.experts.down_proj")
    assert not is_modelopt_quantizable_weight_name("model.layers.0.mlp.experts.gate_up_proj.bias")


def test_get_modelopt_quant_exporter_is_case_insensitive_and_rejects_unknown_modes():
    qformat, export_weight = get_modelopt_quant_exporter("NVFP4")

    assert qformat == QUANTIZATION_NVFP4
    assert export_weight is quantize_nvfp4_weight

    with pytest.raises(ValueError, match="Unsupported ModelOpt quant_mode"):
        get_modelopt_quant_exporter("w4a8")


def test_auto_bridge_modelopt_export_quantizes_matching_weights(monkeypatch):
    conversion_tasks = [
        _task(
            "decoder.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        )
    ]
    bridge = _bridge_for_export(
        conversion_tasks,
        [
            ("model.layers.0.mlp.up_proj.weight", torch.tensor([1.0])),
            ("model.layers.0.mlp.up_proj.bias", torch.tensor([2.0])),
            ("model.layers.0.mlp.up_proj._quantizer._amax", torch.tensor([3.0])),
        ],
    )

    def fake_export_weight(name, tensor, meta):
        assert name == "model.layers.0.mlp.up_proj.weight"
        assert meta.qformat == QUANTIZATION_NVFP4
        yield name, tensor.to(torch.uint8)
        yield "model.layers.0.mlp.up_proj.weight_scale", torch.ones(1)

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {"decoder.layers.0.mlp.up_proj.weight": _quant_meta()},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda quant_mode: (QUANTIZATION_NVFP4, fake_export_weight),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.auto_bridge.model_bridge._get_pp_group",
        lambda _model: None,
    )

    output = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            [object()],
            cpu=True,
            conversion_tasks=conversion_tasks,
        )
    )

    assert [(name, tensor.tolist()) for name, tensor in output] == [
        ("model.layers.0.mlp.up_proj.weight", [1]),
        ("model.layers.0.mlp.up_proj.weight_scale", [1.0]),
        ("model.layers.0.mlp.up_proj.bias", [2.0]),
    ]
    assert bridge.export_calls[0][1]["cpu"] is True
    assert bridge.export_calls[0][1]["conversion_tasks"] is conversion_tasks


def test_auto_bridge_modelopt_export_leaves_ignored_weights_unquantized(monkeypatch):
    conversion_tasks = [
        _task(
            "decoder.layers.0.self_attention.linear_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        )
    ]
    bridge = _bridge_for_export(
        conversion_tasks,
        [("model.layers.0.self_attn.o_proj.weight", torch.tensor([1.0]))],
    )

    def fail_export_weight(*_args, **_kwargs):
        raise AssertionError("ignored weights should not be quantized")

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {"decoder.layers.0.self_attention.linear_proj.weight": _quant_meta()},
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda quant_mode: (QUANTIZATION_NVFP4, fail_export_weight),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.auto_bridge.model_bridge._get_pp_group",
        lambda _model: None,
    )

    output = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            [object()],
            conversion_tasks=conversion_tasks,
            ignore_patterns=["*self_attn*"],
        )
    )

    assert [(name, tensor.tolist()) for name, tensor in output] == [("model.layers.0.self_attn.o_proj.weight", [1.0])]


def test_auto_bridge_modelopt_export_accepts_single_model_and_builds_tasks(monkeypatch):
    model = object()
    conversion_tasks = [
        _task(
            "decoder.embedding.word_embeddings.weight",
            "model.embed_tokens.weight",
        )
    ]
    build_calls = []

    class FakeModelBridge:
        def build_conversion_tasks(self, hf_pretrained, model_arg):
            build_calls.append((hf_pretrained, model_arg))
            return conversion_tasks

    bridge = _bridge_for_export(
        conversion_tasks,
        [("model.embed_tokens.weight", torch.tensor([4.0]))],
    )
    bridge._model_bridge = FakeModelBridge()

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {},
    )

    output = list(
        AutoBridge.export_hf_weights_modelopt(
            bridge,
            model,
            show_progress=False,
            merge_adapter_weights=False,
        )
    )

    assert [(name, tensor.tolist()) for name, tensor in output] == [("model.embed_tokens.weight", [4.0])]
    assert build_calls == [(bridge.hf_pretrained, [model])]
    assert bridge.export_calls[0][0] == [model]
    assert bridge.export_calls[0][1]["show_progress"] is False
    assert bridge.export_calls[0][1]["merge_adapter_weights"] is False
    assert bridge.export_calls[0][1]["conversion_tasks"] is conversion_tasks


def test_auto_bridge_modelopt_export_streams_base_weights_lazily(monkeypatch):
    conversion_tasks = [
        _task(
            "decoder.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        )
    ]
    events = []

    class FakeBridge:
        hf_pretrained = object()
        _model_bridge = SimpleNamespace(
            build_conversion_tasks=lambda *_args, **_kwargs: conversion_tasks,
        )

        def export_hf_weights(self, _model, **_kwargs):
            events.append("start")
            yield "model.layers.0.mlp.up_proj.weight", torch.tensor([1.0])
            events.append("after-first")
            yield "model.layers.0.mlp.down_proj.weight", torch.tensor([2.0])

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {},
    )

    weights = AutoBridge.export_hf_weights_modelopt(
        FakeBridge(),
        [object()],
        conversion_tasks=conversion_tasks,
    )

    assert events == []
    first = next(weights)
    assert first.param_name == "model.layers.0.mlp.up_proj.weight"
    torch.testing.assert_close(first.weight, torch.tensor([1.0]))
    assert events == ["start"]
    second = next(weights)
    assert second.param_name == "model.layers.0.mlp.down_proj.weight"
    torch.testing.assert_close(second.weight, torch.tensor([2.0]))
    assert events == ["start", "after-first"]


def test_auto_bridge_modelopt_export_rejects_mismatched_qformat(monkeypatch):
    conversion_tasks = [
        _task(
            "decoder.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        )
    ]
    bridge = _bridge_for_export(
        conversion_tasks,
        [("model.layers.0.mlp.up_proj.weight", torch.tensor([1.0]))],
    )

    def fail_export_weight(*_args, **_kwargs):
        raise AssertionError("mismatched qformat should fail before quantization")

    monkeypatch.setattr(
        modelopt_utils,
        "collect_modelopt_quant_metadata",
        lambda _tasks: {
            "decoder.layers.0.mlp.up_proj.weight": QuantMeta(
                qformat="unexpected_qformat",
                block_size=16,
                weight_amax=torch.tensor([1.0]),
                weight_scale_2=torch.tensor([1.0]),
            )
        },
    )
    monkeypatch.setattr(
        modelopt_utils,
        "get_modelopt_quant_exporter",
        lambda quant_mode: (QUANTIZATION_NVFP4, fail_export_weight),
    )

    with pytest.raises(RuntimeError, match="Unsupported qformat"):
        list(
            AutoBridge.export_hf_weights_modelopt(
                bridge,
                [object()],
                conversion_tasks=conversion_tasks,
            )
        )
