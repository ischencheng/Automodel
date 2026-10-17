# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""FLUX.3 activation checkpointing covers every transformer block."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel._diffusers.parallelization import Flux3ModelParallelizer, attach_parallelizer
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.model_parallelizer import _parallelize_unsharded_fsdp2
from nemo_automodel.components.distributed.parallelizer import (
    _extract_model_layer_groups,
    _filter_layer_groups_for_activation_checkpointing,
)


class _Block(nn.Module):
    """FLUX.3-style block: attention and feedforward named ``attn`` and ``ff``."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.attn = nn.Linear(4, 4)
        self.ff = nn.Linear(4, 4)


class Flux3Transformer2DModel(nn.Module):
    """Module tree with the block lists of diffusers' Flux3Transformer2DModel."""

    def __init__(self) -> None:
        super().__init__()
        self.txt_mode_blocks = nn.ModuleList([_Block(), _Block()])
        self.mode_blocks = nn.ModuleDict(
            {"image": nn.ModuleList([_Block(), _Block()]), "image_cond": nn.ModuleList([_Block(), _Block()])}
        )
        self.single_transformer_blocks = nn.ModuleList([_Block(), _Block(), _Block()])


def _blocks(model: nn.Module) -> list[nn.Module]:
    return [
        *model.txt_mode_blocks,
        *model.mode_blocks["image"],
        *model.mode_blocks["image_cond"],
        *model.single_transformer_blocks,
    ]


def _checkpoint_unsharded(model: nn.Module) -> nn.Module:
    """Run the single-rank FSDP2 path, which applies activation checkpointing without sharding."""
    mesh_context = SimpleNamespace(
        activation_checkpointing=True, strategy_config=FSDP2Config(), reapply_trainability=None
    )
    return _parallelize_unsharded_fsdp2(model, mesh_context, parallelizer=type(model).parallelizer)


def test_layer_groups_cover_every_block_list() -> None:
    model = Flux3Transformer2DModel()
    attach_parallelizer(model)
    assert isinstance(type(model).parallelizer, Flux3ModelParallelizer)
    layers, _ = _filter_layer_groups_for_activation_checkpointing(_extract_model_layer_groups(model), "all")
    assert [id(layer) for layer in layers] == [
        id(block)
        for block in (
            *model.single_transformer_blocks,
            *model.txt_mode_blocks,
            *model.mode_blocks["image"],
            *model.mode_blocks["image_cond"],
        )
    ]


def test_checkpoints_every_block_as_a_whole() -> None:
    model = Flux3Transformer2DModel()
    attach_parallelizer(model)
    _checkpoint_unsharded(model)
    blocks = _blocks(model)
    assert all(isinstance(block, CheckpointWrapper) for block in blocks)
    # Whole-block wrapping leaves the block's children unwrapped.
    assert not any(isinstance(block._checkpoint_wrapped_module.ff, CheckpointWrapper) for block in blocks)


def test_checkpointed_diffusers_flux3_matches_eager(monkeypatch) -> None:
    """Whole-block checkpointing leaves the output and every gradient of a tiny FLUX.3 transformer unchanged."""
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "Flux3Transformer2DModel"):
        pytest.skip("installed diffusers has no Flux3Transformer2DModel")
    # attach_parallelizer sets class attributes; restore the diffusers class after the test.
    for attr in ("parallelizer", "parallel_layer_groups"):
        monkeypatch.setattr(diffusers.Flux3Transformer2DModel, attr, None, raising=False)

    torch.manual_seed(0)
    eager = diffusers.Flux3Transformer2DModel(
        in_channels=8,
        cond_in_channels=8,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        timestep_channels=16,
        axes_dims_rope=(4, 4, 4, 4),
    )
    checkpointed = diffusers.Flux3Transformer2DModel.from_config(eager.config)
    checkpointed.load_state_dict(eager.state_dict())
    attach_parallelizer(checkpointed)
    _checkpoint_unsharded(checkpointed)
    assert all(isinstance(block, CheckpointWrapper) for block in _blocks(checkpointed))

    image_tokens, text_tokens = 6, 5
    image_ids = torch.zeros(image_tokens, 4)
    image_ids[:, 1] = torch.arange(image_tokens) * 0.5
    text_ids = torch.zeros(text_tokens, 4)
    text_ids[:, 3] = torch.arange(text_tokens)
    inputs = {
        "hidden_states": torch.randn(1, image_tokens, 8),
        "encoder_hidden_states": torch.randn(1, text_tokens, 32),
        "timestep": torch.tensor([0.3]),
        "img_ids": image_ids,
        "txt_ids": text_ids,
    }
    upstream = torch.randn(1, image_tokens, 8)
    outputs = []
    for model in (eager, checkpointed):
        model.train()
        output = model(**inputs, return_dict=False)[0]
        output.backward(upstream)
        outputs.append(output.detach())
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)

    checkpointed_grads = {
        name.replace("._checkpoint_wrapped_module", ""): param.grad for name, param in checkpointed.named_parameters()
    }
    for name, param in eager.named_parameters():
        torch.testing.assert_close(checkpointed_grads[name], param.grad, rtol=0, atol=0, msg=name)
