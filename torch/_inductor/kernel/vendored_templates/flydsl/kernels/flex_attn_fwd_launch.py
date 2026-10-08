# mypy: allow-untyped-defs

import flydsl.compiler as flyc
from flydsl.expr import Stream

from .flex_attn_fwd_helpers import (
    _FOUR_WAVE_PREFILL_MIN_CTAS,
    bf16,
    f32,
    fx,
    i32,
    ir,
    layout,
    llvm,
    slice_view,
    tensor_copy,
    unroll,
    Vec,
    vector,
)


_stream = Stream(None)


def _make_forward_launch(kernel, *, grid, threads, mask_buffer_count, waves_per_eu):
    def dispatch(inputs, mask_buffers, output, stream):
        buffers = (*mask_buffers, *([inputs[5]] * (4 - mask_buffer_count)))
        kernel(*inputs, *buffers, output).launch(
            grid=grid, block=(threads, 1, 1), stream=stream
        )

    # Each JIT entry point exposes only its live mask buffers. The internal
    # placeholder slots alias kv_num_blocks and must remain unused above count.
    if mask_buffer_count == 0:

        @flyc.jit
        def run(
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            output,
            stream: Stream = _stream,
        ):
            dispatch(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (),
                output,
                stream,
            )

    elif mask_buffer_count == 1:

        @flyc.jit
        def run(
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            mask_buffer_0,
            output,
            stream: Stream = _stream,
        ):
            dispatch(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (mask_buffer_0,),
                output,
                stream,
            )

    elif mask_buffer_count == 2:

        @flyc.jit
        def run(
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            mask_buffer_0,
            mask_buffer_1,
            output,
            stream: Stream = _stream,
        ):
            dispatch(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (mask_buffer_0, mask_buffer_1),
                output,
                stream,
            )

    elif mask_buffer_count == 3:

        @flyc.jit
        def run(
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            mask_buffer_0,
            mask_buffer_1,
            mask_buffer_2,
            output,
            stream: Stream = _stream,
        ):
            dispatch(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (mask_buffer_0, mask_buffer_1, mask_buffer_2),
                output,
                stream,
            )

    else:

        @flyc.jit
        def run(
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            mask_buffer_0,
            mask_buffer_1,
            mask_buffer_2,
            mask_buffer_3,
            output,
            stream: Stream = _stream,
        ):
            dispatch(
                (
                    query,
                    key,
                    value,
                    logsumexp,
                    max_scores,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                ),
                (mask_buffer_0, mask_buffer_1, mask_buffer_2, mask_buffer_3),
                output,
                stream,
            )

    run.compile_hints = {"waves_per_eu": waves_per_eu}
    return run


def _select_owner_waves(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q: int,
    seq_kv: int,
    qk_head_dim: int,
) -> int:
    if 0 < seq_q < 128:
        packed_rows = (num_q_heads // num_kv_heads) * seq_q
        required_waves = max(1, (packed_rows + 31) // 32)
        return min(8, 1 << (required_waves - 1).bit_length())

    base_ctas = batch_size * num_q_heads * (seq_q // 128)
    if (
        qk_head_dim == 128
        and base_ctas < _FOUR_WAVE_PREFILL_MIN_CTAS
        and (seq_kv <= 1024 or base_ctas < 256)
    ):
        return 2
    return 4


def _store_output_fragments(
    output_values,
    inverse_vector,
    output_row,
    lane_half,
    output_copy,
    output_chunks,
    mma_tile_size,
):
    for d_chunk in unroll(output_chunks):
        normalized = Vec(output_values[d_chunk]) * inverse_vector
        for column_group in unroll(2):
            packed = (
                vector(
                    [
                        normalized[column_group * 8 + element_index]
                        for element_index in unroll(8)
                    ],
                    f32,
                )
                .to(bf16)
                .bitcast(i32)
            )
            lower = []
            upper = []
            for element_index in unroll(2):
                lhs = i32(packed[element_index]).ir_value()
                rhs = i32(packed[element_index + 2]).ir_value()
                swapped = fx.rocdl.permlane32_swap(
                    ir.Type.parse("!llvm.struct<(i32, i32)>"), lhs, rhs, False, True
                )
                lower.append(i32(llvm.extractvalue(i32.ir_type, swapped, [0])))
                upper.append(i32(llvm.extractvalue(i32.ir_type, swapped, [1])))
            values = vector([*lower, *upper], i32).bitcast(bf16)
            column = i32(d_chunk * mma_tile_size + column_group * 16) + lane_half * i32(
                8
            )
            fragment = fx.make_rmem_tensor(8, bf16)
            fragment.store(values)
            tensor_copy(
                output_copy, fragment, slice_view(output_row, (None, column // i32(8)))
            )
            fx.rocdl.sched_barrier(0)


def make_forward_shared_memory(
    pipelined,
    split_kv,
    kv_tile_rows,
    qk_head_dim,
    value_head_dim,
    query_rows,
    waves,
    output_chunks,
):
    if pipelined:

        @fx.struct
        class ForwardSharedMemory:
            # Keep Q in registers and double-buffer K/V
            # so the next tile's DMA can overlap the current tile's math.
            key_stage_0: fx.Array[bf16, kv_tile_rows * qk_head_dim, 16]
            key_stage_1: fx.Array[bf16, kv_tile_rows * qk_head_dim, 16]
            value_stage_0: fx.Array[bf16, kv_tile_rows * value_head_dim, 16]
            value_stage_1: fx.Array[bf16, kv_tile_rows * value_head_dim, 16]

    elif split_kv:

        @fx.struct
        class ForwardSharedMemory:
            query: fx.Array[bf16, query_rows * qk_head_dim, 16]
            # One reusable K/V tile per worker wave keeps the CTA below 64 KiB.
            key_value: fx.Array[
                bf16,
                waves * kv_tile_rows * qk_head_dim,
                16,
            ]
            reduction_stats: fx.Array[f32, 2 * 2, 16]
            reduction_output: fx.Array[f32, 2 * output_chunks * 16, 16]

    else:

        @fx.struct
        class ForwardSharedMemory:
            query: fx.Array[bf16, query_rows * qk_head_dim, 16]
            # K needs the largest allocation. V reuses the same storage after
            # every wave has consumed K into registers.
            key_value: fx.Array[bf16, kv_tile_rows * qk_head_dim, 16]

    return ForwardSharedMemory
