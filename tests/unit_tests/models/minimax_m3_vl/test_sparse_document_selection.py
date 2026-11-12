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

"""Generic MiniMax M3 sparse selection on packed documents (CPU).

A packed document must select the key blocks it selects when it runs alone, which is
also what MSA's document-local rule gives.
"""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import MiniMaxM3CPSparseAttention
from nemo_automodel.components.models.minimax_m3_vl.layers import (
    select_document_sparse_blocks,
    select_sparse_blocks,
    sparse_document_layout,
)
from nemo_automodel.components.models.minimax_m3_vl.msa import MSAMicrobatch
from tests.unit_tests.models.minimax_m3_vl._msa_select_reference import select_blocks_reference
from tests.unit_tests.models.minimax_m3_vl.test_sparse_replay import _cpu_attention

LENGTHS = [13, 22, 9, 17]


def _packed_inputs(lengths):
    """Return per-document position ids [1, T] and the document keep-mask [1, 1, T, T]."""
    total = sum(lengths)
    positions = torch.cat([torch.arange(n) for n in lengths])[None]
    documents = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])
    keep = (documents[:, None] == documents[None, :]) & torch.tril(torch.ones(total, total, dtype=torch.bool))
    return positions, keep[None, None]


def test_packed_documents_match_each_document_alone(sparse_model):
    """Every document of a packed row gets the hidden states it gets when it runs alone."""
    generator = torch.Generator().manual_seed(1)
    docs = [torch.randint(0, 100, (n,), generator=generator) for n in LENGTHS]
    positions, keep = _packed_inputs(LENGTHS)
    with torch.no_grad():
        packed = sparse_model.model(torch.cat(docs)[None], position_ids=positions, attention_mask=keep)
        start = 0
        for doc in docs:
            alone = sparse_model.model(doc[None])
            torch.testing.assert_close(packed[0, start : start + doc.numel()], alone[0], rtol=1e-5, atol=1e-5)
            start += doc.numel()


def test_local_sparse_path_selects_per_document(sparse_model, monkeypatch):
    """The CUDA local path (CPU attention stand-in) matches each document run alone."""
    monkeypatch.setattr(MiniMaxM3CPSparseAttention, "_flex_sparse_attention", _cpu_attention)
    attn = sparse_model.model.layers["1"].self_attn
    assert isinstance(attn, MiniMaxM3CPSparseAttention)
    generator = torch.Generator().manual_seed(2)
    hidden = torch.randn(1, sum(LENGTHS), sparse_model.config.hidden_size, generator=generator)
    positions, keep = _packed_inputs(LENGTHS)
    layout = sparse_document_layout(positions, block_size=attn.indexer.block_size)
    with torch.no_grad():
        freqs = sparse_model.model.make_freqs_cis(positions)
        packed = attn._local_sparse_forward(hidden, freqs_cis=freqs, attention_mask=keep, documents=layout)
        row_selected = attn._local_sparse_forward(hidden, freqs_cis=freqs, attention_mask=keep)
        start, differs = 0, False
        for n in LENGTHS:
            doc = hidden[:, start : start + n]
            alone = attn._local_sparse_forward(
                doc, freqs_cis=sparse_model.model.make_freqs_cis(torch.arange(n)[None]), attention_mask=None
            )
            torch.testing.assert_close(packed[:, start : start + n], alone, rtol=1e-5, atol=1e-5)
            differs |= not torch.allclose(row_selected[:, start : start + n], alone, rtol=1e-5, atol=1e-5)
            start += n
    assert differs


@pytest.mark.parametrize("score_type", ["max", "lse"])
@pytest.mark.parametrize("init_blocks", [0, 1])
def test_document_selection_matches_msa_reference(score_type, init_blocks):
    """Selected blocks, in document-local ids, equal MSA's reference rule."""
    block_size, topk = 4, 2
    total = sum(LENGTHS)
    generator = torch.Generator().manual_seed(3)
    idx_q = torch.randn(1, total, 2, 8, generator=generator)
    idx_k = torch.randn(1, total, 1, 8, generator=generator)
    positions, _ = _packed_inputs(LENGTHS)
    layout = sparse_document_layout(positions, block_size=block_size)
    selected = select_document_sparse_blocks(
        idx_q,
        idx_k,
        layout=layout,
        topk_blocks=topk,
        init_blocks=init_blocks,
        local_blocks=1,
        score_type=score_type,
    )[0]  # [index_heads, T, workspace_blocks]

    doc_map = torch.cat([torch.full((n,), i + 1) for i, n in enumerate(LENGTHS)])[None]
    microbatch = MSAMicrobatch.from_document_map(doc_map, forced_blocks=(init_blocks, 1))
    reference = select_blocks_reference(
        microbatch,
        idx_q[0],
        idx_k[0],
        block_size=block_size,
        topk_blocks=topk,
        init_blocks=init_blocks,
        local_blocks=1,
        score_type=score_type,
    )  # [index_heads, T, topk] document-local ids, -1 padded
    first = layout.first_blocks[0]
    for head in range(selected.shape[0]):
        for query in range(total):
            ours = {int(b) - int(first[query]) for b in selected[head, query].nonzero().flatten()}
            theirs = {int(b) for b in reference[head, query] if b >= 0}
            assert ours == theirs, (head, query, ours, theirs)


def test_query_subset_matches_full_rows():
    """Context parallelism selects for a subset of query slots against all keys."""
    total = sum(LENGTHS)
    generator = torch.Generator().manual_seed(5)
    idx_q = torch.randn(1, total, 2, 8, generator=generator)
    idx_k = torch.randn(1, total, 1, 8, generator=generator)
    positions, _ = _packed_inputs(LENGTHS)
    layout = sparse_document_layout(positions, block_size=4)
    kwargs = dict(layout=layout, topk_blocks=2, init_blocks=0, local_blocks=1)
    full = select_document_sparse_blocks(idx_q, idx_k, **kwargs)
    slots = torch.tensor([0, 5, 12, 13, 30, 34, 35, 44, 60])
    subset = select_document_sparse_blocks(idx_q[:, slots], idx_k, q_slots=slots, **kwargs)
    assert torch.equal(subset, full[:, :, slots])


def test_layout_aligns_documents_and_skips_padding():
    """Documents start on block boundaries; zero-padded positions stay in the last document."""
    positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 0, 0]])
    padding = torch.tensor([[False] * 10 + [True] * 3])
    layout = sparse_document_layout(positions, block_size=4, padding_mask=padding)
    assert layout.key_rows[0].tolist() == [0, 1, 2, 3, 4, 8, 9, 12, 13, 14, 15, 16, 17]
    assert layout.first_blocks[0].tolist() == [0] * 5 + [2] * 2 + [3] * 6
    assert layout.num_blocks == 5
    # Without a padding mask, repeated zeros still open only one document.
    unmasked = sparse_document_layout(positions.clone(), block_size=4)
    assert unmasked.key_rows[0].tolist() == [0, 1, 2, 3, 4, 8, 9, 12, 13, 14, 16, 17, 18]


def test_single_document_rows_keep_the_row_layout():
    """Unpacked rows get no layout, and a zero first block selects as before."""
    assert sparse_document_layout(torch.arange(20)[None], block_size=4) is None
    generator = torch.Generator().manual_seed(7)
    idx_q = torch.randn(2, 20, 2, 8, generator=generator)
    idx_k = torch.randn(2, 20, 1, 8, generator=generator)
    kwargs = dict(block_size=4, topk_blocks=2, init_blocks=1, local_blocks=1)
    plain = select_sparse_blocks(idx_q, idx_k, **kwargs)
    zero_first = select_sparse_blocks(idx_q, idx_k, q_first_blocks=torch.zeros(20, dtype=torch.long), **kwargs)
    assert torch.equal(plain, zero_first)


def test_packed_selection_differs_from_row_selection():
    """Guard the fixture: with these lengths the row-level selection does cross documents."""
    total = sum(LENGTHS)
    generator = torch.Generator().manual_seed(3)
    idx_q = torch.randn(1, total, 2, 8, generator=generator)
    idx_k = torch.randn(1, total, 1, 8, generator=generator)
    positions, keep = _packed_inputs(LENGTHS)
    layout = sparse_document_layout(positions, block_size=4)
    per_document = select_document_sparse_blocks(
        idx_q, idx_k, layout=layout, topk_blocks=2, init_blocks=0, local_blocks=1
    )
    per_document_keys = per_document.gather(-1, layout.key_blocks[:, None, None, :].expand(1, 2, total, total))
    row = select_sparse_blocks(idx_q, idx_k, block_size=4, topk_blocks=2, init_blocks=0, local_blocks=1)
    row_keys = row.repeat_interleave(4, dim=-1)[..., :total]
    causal = keep[0, 0]
    assert not torch.equal(per_document_keys & causal, row_keys & causal)
