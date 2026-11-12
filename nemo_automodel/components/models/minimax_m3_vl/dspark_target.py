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

"""Packed DSpark target forward for MiniMax M3."""

from __future__ import annotations

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.dspark.common import context_doc_ids


def dspark_packed_forward(model: nn.Module, input_ids: torch.Tensor, *, seq_lens: torch.Tensor) -> None:
    """Run a frozen MiniMax M3 forward on packed rows for DSpark feature capture.

    M3 takes packed documents as a boolean ``[batch, 1, sequence, sequence]`` document
    keep-mask with per-document ``position_ids``; its sparse layers then select key
    blocks inside each document. The rows keep their layout, and only the last
    position's logits are projected since DSpark reads hooked activations.

    Args:
        model: ``MiniMaxM3SparseForConditionalGeneration`` target.
        input_ids: Token IDs of shape [batch, sequence] holding packed documents.
        seq_lens: Document lengths of shape [batch, documents]; each row sums to
            ``sequence`` and zero entries pad the document axis.
    """
    batch, length = input_ids.shape
    device = input_ids.device
    documents = context_doc_ids(seq_lens, length, device)  # [batch, sequence]
    slots = torch.arange(length, device=device).expand(batch, -1)
    first_slot = (
        torch.where(torch.nn.functional.pad(documents[:, 1:] != documents[:, :-1], (1, 0), value=True), slots, 0)
        .cummax(dim=-1)
        .values
    )
    causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    keep = (documents[:, :, None] == documents[:, None, :]) & causal
    model(input_ids, position_ids=slots - first_slot, attention_mask=keep[:, None], logits_to_keep=1)
