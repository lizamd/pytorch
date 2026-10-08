# mypy: allow-untyped-defs

import torch
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE

from ...codegen.flydsl.flydsl_template import FlyDSLTemplate
from ...ir import FixedLayout, Pointwise
from ...lowering import empty_strided, full
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import ops, V
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    freeze_irnodes,
    get_fwd_subgraph_outputs,
    infer_dense_strides,
    load_flex_template,
    maybe_realize,
)
from .flex_flash_attention import is_trivial_mask_graph
from .flex_flydsl_config import _get_flydsl_flex_attention_forward_config


flex_flydsl_forward_template = FlyDSLTemplate(
    name="flex_flydsl_forward", source=load_flex_template("flydsl_forward")
)


def maybe_append_flydsl_flex_attention_choice(
    choices,
    *,
    query,
    key,
    value,
    logsumexp,
    max_scores,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    layout,
    write_max_scores=True,
    **config_inputs,
) -> tuple[bool, str]:
    config, reason = _get_flydsl_flex_attention_forward_config(
        query=query,
        key=key,
        value=value,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        **config_inputs,
    )
    if config is None:
        return False, reason

    config["WRITE_MAX_SCORES"] = bool(write_max_scores)

    input_nodes = [
        query,
        key,
        value,
        logsumexp,
        max_scores,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ]
    mask_mod_other_buffers = config_inputs["mask_mod_other_buffers"]
    mask_buffer_count = config["MASK_BUFFER_COUNT"]
    if mask_buffer_count:
        if len(mask_mod_other_buffers) != mask_buffer_count:
            return False, "mask_mod capture count changed during lowering"
        input_nodes.extend(mask_mod_other_buffers)

    choices_before = len(choices)
    error = flex_flydsl_forward_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        mutated_inputs=[logsumexp, max_scores],
        layout=layout,
        **config,
    )
    if len(choices) == choices_before:
        return False, f"FlyDSL template registration failed: {error}"
    return True, ""


def _create_dense_metadata(query, key):
    """Materialize only the forward metadata for the frontend's no-mask sentinel."""
    seq_q = V.graph.sizevars.guard_int(query.get_size()[2])
    seq_kv = V.graph.sizevars.guard_int(key.get_size()[2])
    num_q_blocks = (seq_q + 127) // 128
    num_kv_blocks = (seq_kv + 127) // 128
    shape = [1, 1, num_q_blocks]
    device = query.get_device()
    return (
        full(shape, 0, dtype=torch.int32, device=device),
        full([*shape, 1], 0, dtype=torch.int32, device=device),
        full(shape, num_kv_blocks, dtype=torch.int32, device=device),
        Pointwise.create(
            device=device,
            dtype=torch.int32,
            ranges=[*shape, num_kv_blocks],
            inner_fn=lambda index: ops.index_expr(index[-1], torch.int32),
        ),
    )


def create_flydsl_flex_attention_kernel(
    *,
    query,
    key,
    value,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    subgraph_buffer,
    mask_graph_buffer,
    write_max_scores=True,
    **config_inputs,
):
    """Lower the explicitly selected FlyDSL backend independently of Triton."""
    score_buffers = maybe_realize(config_inputs["score_mod_other_buffers"])
    mask_buffers = maybe_realize(config_inputs["mask_mod_other_buffers"])
    sparse_q_block_size = config_inputs["sparse_q_block_size"]
    sparse_kv_block_size = config_inputs["sparse_kv_block_size"]
    if (
        sparse_q_block_size == _LARGE_SPARSE_BLOCK_SIZE
        and sparse_kv_block_size == _LARGE_SPARSE_BLOCK_SIZE
        and is_trivial_mask_graph(config_inputs["mask_graph"].graph_module)
    ):
        (kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices) = (
            _create_dense_metadata(query, key)
        )
        sparse_q_block_size = sparse_kv_block_size = 128

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
    )
    freeze_irnodes(score_buffers)
    freeze_irnodes(mask_buffers)

    batch, heads, seq_q, _ = query.get_size()
    output_size = [batch, heads, seq_q, value.get_size()[-1]]
    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        output_size,
        stride=infer_dense_strides(output_size, query.get_stride()),
    )
    logsumexp = empty_strided(
        [batch, heads, seq_q], None, dtype=torch.float32, device=query.get_device()
    )
    max_scores = empty_strided(
        [batch, heads, seq_q], None, dtype=torch.float32, device=query.get_device()
    )
    config_inputs.update(
        score_mod_other_buffers=score_buffers,
        mask_mod_other_buffers=mask_buffers,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
    )
    inputs = [
        query,
        key,
        value,
        logsumexp,
        max_scores,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        *mask_buffers,
    ]
    choices = []
    appended, reason = maybe_append_flydsl_flex_attention_choice(
        choices,
        query=query,
        key=key,
        value=value,
        logsumexp=logsumexp,
        max_scores=max_scores,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        layout=layout,
        write_max_scores=write_max_scores,
        **config_inputs,
    )
    if not appended:
        raise RuntimeError(
            "BACKEND='FLYDSL' but the FlyDSL flex forward candidate "
            f"could not be registered: {reason}"
        )
    output, _ = autotune_select_algorithm(
        "flex_attention_flydsl",
        choices,
        inputs,
        layout,
        input_gen_fns={
            5: create_num_blocks_fake_generator(kv_indices),
            6: create_indices_fake,
            7: create_num_blocks_fake_generator(full_kv_indices),
            8: create_indices_fake,
        },
    )
    output.data.data.subgraph_inps = list(score_buffers) + list(mask_buffers)
    output.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )
    return output, logsumexp, max_scores
