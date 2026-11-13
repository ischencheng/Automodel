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

"""Inkling random initialization through the registered VL factory and the real Checkpointer.

``InklingForConditionalGeneration`` skips random init before its own checkpoint load. ``from_config`` has no
load, so every parameter must still be initialized. AutoModel's CUDA device query is adapted to CPU and the
materialized storage is poisoned to expose a skipped initializer; construction and initialization stay real.
"""

from unittest.mock import patch

import torch
import torch.nn.functional as F

from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components.checkpoint import checkpointing
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration

from .parity_check_inkling import build_tiny_config


def test_from_config_initializes_every_parameter() -> None:
    """A from_config model has finite parameters, logits and gradients."""
    assert InklingForConditionalGeneration._skip_init_weights_on_load is True
    materialize = checkpointing.to_empty_parameters_only

    def materialize_and_poison(model: torch.nn.Module, *, device: torch.device) -> None:
        materialize(model, device=device)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float("nan"))

    config = build_tiny_config()
    config.architectures = ["InklingForConditionalGeneration"]
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")),
        patch.object(checkpointing, "to_empty_parameters_only", side_effect=materialize_and_poison) as materialized,
    ):
        model = NeMoAutoModelForImageTextToText.from_config(
            config,
            backend=backend,
            dtype=torch.float32,
            attn_implementation="sdpa",
            force_hf=False,
            use_liger_kernel=False,
            use_sdpa_patching=False,
        )
    assert isinstance(model, InklingForConditionalGeneration)
    materialized.assert_called_once()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name

    model.train()
    input_ids = torch.arange(1, 13).unsqueeze(0)
    logits = model(input_ids=input_ids)
    logits = getattr(logits, "logits", logits)
    assert torch.isfinite(logits).all()
    F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), input_ids[:, 1:].flatten()).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
