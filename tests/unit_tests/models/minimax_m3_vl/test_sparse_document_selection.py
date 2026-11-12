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
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

import nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn as cp
from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import MiniMaxM3CPSparseAttention
from nemo_automodel.components.models.minimax_m3_vl.layers import (
    packed_sparse_documents,
    select_document_sparse_blocks,
    select_sparse_blocks,
    sparse_document_layout,
)
from nemo_automodel.components.models.minimax_m3_vl.msa import MSAMicrobatch
from tests.unit_tests.models.minimax_m3_vl._msa_select_reference import select_blocks_reference
from tests.unit_tests.models.minimax_m3_vl.test_sparse_replay import _block, _cpu_attention

LENGTHS = [13, 22, 9, 17]
# Two equal-width rows with different layouts; the second row holds one document.
ROWS = ([13, 22, 9, 17], [61])


def _packed_inputs(lengths):
    """Return per-document position ids [1, T] and the document keep-mask [1, 1, T, T]."""
    total = sum(lengths)
    positions = torch.cat([torch.arange(n) for n in lengths])[None]
    documents = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])
    keep = (documents[:, None] == documents[None, :]) & torch.tril(torch.ones(total, total, dtype=torch.bool))
    return positions, keep[None, None]


def _packed_rows(rows):
    """Stack per-row position ids [B, T] and document keep-masks [B, 1, T, T]."""
    inputs = [_packed_inputs(row) for row in rows]
    return torch.cat([p for p, _ in inputs]), torch.cat([k for _, k in inputs])


@pytest.mark.parametrize("rows", [(LENGTHS,), ROWS], ids=["one_row", "two_rows"])
def test_packed_documents_match_each_document_alone(sparse_model, rows):
    """Every document of a packed row gets the hidden states it gets when it runs alone."""
    generator = torch.Generator().manual_seed(1)
    docs = [[torch.randint(0, 100, (n,), generator=generator) for n in row] for row in rows]
    positions, keep = _packed_rows(rows)
    input_ids = torch.stack([torch.cat(row) for row in docs])
    with torch.no_grad():
        packed = sparse_model.model(input_ids, position_ids=positions, attention_mask=keep)
        for b, row in enumerate(docs):
            start = 0
            for doc in row:
                alone = sparse_model.model(doc[None])
                torch.testing.assert_close(packed[b, start : start + doc.numel()], alone[0], rtol=1e-5, atol=1e-5)
                start += doc.numel()


def test_local_sparse_path_selects_per_document(sparse_model, monkeypatch):
    """The CUDA local path (CPU attention stand-in) matches each document run alone."""
    monkeypatch.setattr(MiniMaxM3CPSparseAttention, "_flex_sparse_attention", _cpu_attention)
    attn = sparse_model.model.layers["1"].self_attn
    assert isinstance(attn, MiniMaxM3CPSparseAttention)
    generator = torch.Generator().manual_seed(2)
    hidden = torch.randn(len(ROWS), sum(ROWS[0]), sparse_model.config.hidden_size, generator=generator)
    positions, keep = _packed_rows(ROWS)
    layout = sparse_document_layout(positions, block_size=attn.indexer.block_size)
    with torch.no_grad():
        freqs = sparse_model.model.make_freqs_cis(positions)
        packed = attn._local_sparse_forward(hidden, freqs_cis=freqs, attention_mask=keep, documents=layout)
        row_selected = attn._local_sparse_forward(hidden, freqs_cis=freqs, attention_mask=keep)
        differs = False
        for b, row in enumerate(ROWS):
            start = 0
            for n in row:
                doc = hidden[b : b + 1, start : start + n]
                alone = attn._local_sparse_forward(
                    doc, freqs_cis=sparse_model.model.make_freqs_cis(torch.arange(n)[None]), attention_mask=None
                )
                torch.testing.assert_close(packed[b : b + 1, start : start + n], alone, rtol=1e-5, atol=1e-5)
                differs |= not torch.allclose(row_selected[b : b + 1, start : start + n], alone, rtol=1e-5, atol=1e-5)
                start += n
    assert differs


def _cpu_packed_forward(
    self: MiniMaxM3CPSparseAttention,
    x: torch.Tensor,
    *,
    freqs_cis: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Enter the local path on CPU the way the CUDA dispatch does for a packed batch."""
    documents = packed_sparse_documents(attention_mask, position_ids, padding_mask, block_size=self.indexer.block_size)
    return self._local_sparse_forward(x, freqs_cis=freqs_cis, attention_mask=attention_mask, documents=documents)


@pytest.mark.parametrize("selective", [False, True])
def test_checkpoint_recompute_replays_document_selection(monkeypatch, selective):
    """Activation checkpointing replays the per-document selection and matches gradients.

    The checkpointed call is the first to see its position ids, so it builds the layout inside
    the checkpointed region and its recompute finds the memo; the selective variant saves
    matmuls as a selective policy would.
    """
    monkeypatch.setattr(MiniMaxM3CPSparseAttention, "forward", _cpu_packed_forward)
    monkeypatch.setattr(MiniMaxM3CPSparseAttention, "_flex_sparse_attention", _cpu_attention)
    calls = []
    original = cp.select_document_sparse_blocks

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cp, "select_document_sparse_blocks", count)
    torch.manual_seed(5)
    block = _block()
    lengths = [7, 10]
    positions, keep = _packed_inputs(lengths)
    angles = positions[0, :, None].float() / (10000 ** (torch.arange(0, 8, 2) / 8))
    freqs = torch.cat((angles.cos(), angles.sin()), -1).unsqueeze(0)
    x = torch.randn(1, sum(lengths), 32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    expected = block(xr, freqs_cis=freqs, attention_mask=keep, position_ids=positions.clone())
    assert len(calls) == 1

    def policy(ctx, op, *args, **kwargs):
        if op in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    inner = (lambda: create_selective_checkpoint_contexts(policy)) if selective else None
    actual = checkpoint(
        block,
        x,
        use_reentrant=False,
        context_fn=block.nemo_checkpoint_context_fn(inner),
        freqs_cis=freqs,
        attention_mask=keep,
        position_ids=positions.clone(),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    assert len(calls) == 2, "the recompute must replay the recorded selection"
    torch.testing.assert_close(x.grad, xr.grad, rtol=2e-5, atol=2e-6)


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
    # Without a padding mask, a run of zeros opens one document.
    unmasked = sparse_document_layout(positions.clone(), block_size=4)
    assert unmasked.key_rows[0].tolist() == [0, 1, 2, 3, 4, 8, 9, 12, 13, 14, 16, 17, 18]
    # With one, a one-token document is a document of its own.
    single = sparse_document_layout(
        torch.tensor([[0, 1, 2, 0, 0, 1, 2]]), block_size=4, padding_mask=torch.zeros(1, 7, dtype=torch.bool)
    )
    assert single.key_rows[0].tolist() == [0, 1, 2, 4, 8, 9, 10]
    assert single.first_blocks[0].tolist() == [0, 0, 0, 1, 2, 2, 2]


def test_layout_is_built_outside_dispatch_modes():
    """Selective checkpointing never sees the one-time layout ops (on CUDA, a saved host copy), so a
    recompute that finds the memo replays the block's other saved ops in order."""
    seen = []

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(func)
            return func(*args, **(kwargs or {}))

    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
    with Record():
        layout = sparse_document_layout(positions, block_size=4)
    assert layout is not None
    assert seen == []


def test_layout_memo_follows_in_place_updates():
    """A position tensor rewritten in place gets a new layout."""
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    first = sparse_document_layout(positions, block_size=4)
    assert sparse_document_layout(positions, block_size=4) is first
    positions.copy_(torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]))
    assert sparse_document_layout(positions, block_size=4).key_rows[0].tolist() == [0, 1, 2, 4, 5, 6, 7, 8]


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
