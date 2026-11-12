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

"""DSpark sequence packing for MiniMax M3 targets and drafts (CPU)."""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl import dspark_target
from nemo_automodel.components.models.minimax_m3_vl.dspark import MiniMaxM3DSparkModel
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.recipes.speculative_decoding.train_dspark import _validate_draft_packing
from tests.unit_tests.models.minimax_m3_vl.test_minimax_m3_dspark import (
    HIDDEN,
    TARGET_LAYER_IDS,
    VOCAB,
    _build_minimax_m3_draft,
    _build_released_draft,
)

# Two equal-width rows; documents start mid-block for the tiny target's 4-token sparse blocks.
ROW_DOCUMENTS = ([13, 22, 9], [19, 25])
PARITY_TOL = 2e-5


def _pack(row_documents, vocab_size: int, generator: torch.Generator):
    """Pack random documents into equal-width rows.

    Returns:
        Per-row lists of document token tensors of shape [1, length], and the packed
        input_ids, position_ids, doc_remaining of shape [batch, sequence] and
        zero-padded seq_lens of shape [batch, max_documents].
    """
    documents = [[torch.randint(1, vocab_size, (1, n), generator=generator) for n in row] for row in row_documents]
    input_ids = torch.cat([torch.cat(row, dim=1) for row in documents], dim=0)
    position_ids = torch.stack([torch.cat([torch.arange(n) for n in row]) for row in row_documents])
    doc_remaining = torch.stack([torch.cat([torch.arange(n - 1, -1, -1) for n in row]) for row in row_documents])
    max_documents = max(len(row) for row in row_documents)
    seq_lens = torch.tensor([row + [0] * (max_documents - len(row)) for row in row_documents])
    return documents, input_ids, position_ids, doc_remaining, seq_lens


def test_packed_target_features_match_each_document_alone(vlm_model):
    """Captured features and final hidden states of a packed row equal each document run alone."""
    wrapper = HFDSparkTargetModel(
        vlm_model, target_layer_ids=[0, 2], packed_forward=dspark_target.dspark_packed_forward
    )
    documents, input_ids, position_ids, doc_remaining, seq_lens = _pack(
        ROW_DOCUMENTS, vocab_size=100, generator=torch.Generator().manual_seed(1)
    )
    ones = torch.ones_like(input_ids)
    with torch.no_grad():
        packed = wrapper.generate_batch(
            input_ids, ones, ones, position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining
        )
        for row, row_documents in enumerate(documents):
            alone = [wrapper.generate_batch(doc, torch.ones_like(doc), torch.ones_like(doc)) for doc in row_documents]
            torch.testing.assert_close(
                packed.target_hidden_states[row : row + 1],
                torch.cat([batch.target_hidden_states for batch in alone], dim=1),
                rtol=PARITY_TOL,
                atol=PARITY_TOL,
            )
            torch.testing.assert_close(
                packed.target_last_hidden_states[row : row + 1],
                torch.cat([batch.target_last_hidden_states for batch in alone], dim=1),
                rtol=PARITY_TOL,
                atol=PARITY_TOL,
            )


def test_generic_packing_mask_is_rejected_by_the_target(vlm_model):
    """Without the model-owned forward, the wrapper's additive block-causal mask cannot reach M3."""
    wrapper = HFDSparkTargetModel(vlm_model, target_layer_ids=[0, 2])
    _, input_ids, position_ids, doc_remaining, seq_lens = _pack(
        ROW_DOCUMENTS, vocab_size=100, generator=torch.Generator().manual_seed(1)
    )
    ones = torch.ones_like(input_ids)
    with torch.no_grad(), pytest.raises(ValueError, match="Additive float masks are not supported"):
        wrapper.generate_batch(
            input_ids, ones, ones, position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining
        )


@pytest.mark.parametrize("released", [False, True], ids=["native", "released_window"])
@pytest.mark.parametrize("attn_implementation", ["sdpa", "flex_attention"])
def test_packed_draft_blocks_match_the_document_alone(tmp_path, released, attn_implementation):
    """Blocks anchored in a document match the unpacked run of that document alone.

    The row packs an unsupervised document, the document under test, and a supervised
    document. The tested document is longer than the released 12-token window, so the
    window applies inside it. num_anchors exceeds the candidate count, so both runs
    sample every candidate anchor.
    """
    if released:
        draft = _build_released_draft(tmp_path, attn_implementation)
    else:
        draft = _build_minimax_m3_draft()
        draft.config._attn_implementation = attn_implementation
    draft.num_anchors = 64
    lengths = [9, 20, 11]
    start, end = lengths[0], lengths[0] + lengths[1]
    generator = torch.Generator().manual_seed(3)
    _, input_ids, position_ids, doc_remaining, seq_lens = _pack([lengths], vocab_size=VOCAB - 1, generator=generator)
    features = torch.randn(1, sum(lengths), len(TARGET_LAYER_IDS) * HIDDEN, generator=generator)
    last_hidden = torch.randn(1, sum(lengths), HIDDEN, generator=generator)
    loss_mask = torch.ones(1, sum(lengths))
    loss_mask[:, :start] = 0

    with torch.no_grad():
        alone = draft(
            input_ids[:, start:end], features[:, start:end], loss_mask[:, start:end], last_hidden[:, start:end]
        )
        packed = draft(
            input_ids,
            features,
            loss_mask,
            last_hidden,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )

    # The tested document yields the first anchors of the packed row, in the same order.
    kept = int(alone.block_keep_mask.sum())
    assert kept == lengths[1] - 1
    assert int(packed.block_keep_mask.sum()) == kept + lengths[2] - 1
    eval_mask = alone.eval_mask[:, :kept]
    assert eval_mask.any()
    torch.testing.assert_close(packed.eval_mask[:, :kept], eval_mask)
    torch.testing.assert_close(packed.target_ids[:, :kept][eval_mask], alone.target_ids[:, :kept][eval_mask])
    for field in ("draft_logits", "confidence_pred", "aligned_target_logits"):
        torch.testing.assert_close(
            getattr(packed, field)[:, :kept][eval_mask],
            getattr(alone, field)[:, :kept][eval_mask],
            atol=PARITY_TOL,
            rtol=PARITY_TOL,
        )


def test_minimax_m3_draft_accepts_packing():
    assert MiniMaxM3DSparkModel.supports_sequence_packing is True
    _validate_draft_packing(_build_minimax_m3_draft())
