# mypy: allow-untyped-defs

from .flex_attn_fwd_helpers import (
    _exp2,
    _f32,
    _FOUR_WAVE_PREFILL_MIN_CTAS,
    _LN2,
    _LOG2E,
    _maximum,
    _NEG_BIG,
    _select_waves_per_eu,
    bf16,
    const_expr,
    divide,
    f32,
    flyc,
    fx,
    i32,
    i64,
    ir,
    layout,
    llvm,
    make_global_view,
    make_kv_staging,
    make_mask_buffers,
    make_mask_evaluator,
    make_qk_shared_layout,
    make_tile_processor,
    make_value_shared_layout,
    scalar,
    slice_view,
    tensor_copy,
    tensor_view,
    u32,
    unroll,
    Vec,
    vector,
)
from .flex_attn_fwd_launch import (
    _make_forward_launch,
    _select_owner_waves,
    _store_output_fragments,
    make_forward_shared_memory,
)
from .flex_attn_fwd_metadata import make_paired_metadata_ops
from .flex_attn_fwd_pipeline import make_staged_pipeline, reduce_split_results
from .flex_attn_utils import analyze_mask_access


def build_flex_attn_fwd_module(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q: int,
    seq_kv: int,
    qk_head_dim: int,
    v_head_dim: int,
    block_mask_batch: int,
    block_mask_heads: int,
    num_q_blocks: int,
    max_partial_blocks: int,
    max_full_blocks: int,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    scale: float,
    mask_program=(),
    mask_program_output: int = 0,
    mask_buffer_shapes=(),
    mask_buffer_strides=(),
    q_stride=None,
    k_stride=None,
    v_stride=None,
    o_stride=None,
    output_stats_in_log2: bool = False,
    write_max_scores: bool = True,
):
    """Build the gfx950 prefill or packed-GQA decode kernel.

    Each wave is an independent owner of 32 query rows and its corresponding
    32x128 output tile. The owner count is selected at compile time; K/V are
    staged once per CTA and shared by all owners.
    """

    batch_count, query_heads, kv_heads = batch_size, num_q_heads, num_kv_heads
    qk_head_dim, value_head_dim = qk_head_dim, v_head_dim
    sparse_query_size, sparse_kv_size = sparse_q_block_size, sparse_kv_block_size
    mask_batches, mask_heads = block_mask_batch, block_mask_heads
    partial_capacity, full_capacity = max_partial_blocks, max_full_blocks
    mask_output_slot = mask_program_output
    stats_in_log2 = output_stats_in_log2

    # Keep standalone entry-point validation even though Inductor checks these
    # constraints before registering the vendored kernel.
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("FlyDSL forward requires Hq % Hkv == 0")

    paired_ctas = batch_count * query_heads * (seq_q // 256)
    min_kv_blocks = 64 if mask_buffer_shapes else 16
    # Pair sparse rows only when enough CTAs and KV work amortize the merge.
    paired = (
        seq_q % 256 == 0
        and sparse_query_size == 128
        and sparse_kv_size == 128
        and paired_ctas >= _FOUR_WAVE_PREFILL_MIN_CTAS
        and seq_kv // sparse_kv_size >= min_kv_blocks
    )

    decode = 0 < seq_q < 128
    owner_waves = (
        8
        if paired
        else _select_owner_waves(
            batch_size=batch_count,
            num_q_heads=query_heads,
            num_kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_head_dim=qk_head_dim,
        )
    )
    query_rows = owner_waves * 32
    kv_tile_rows = 64
    split_kv = (
        (query_heads // kv_heads) * seq_q == 1
        and batch_count * kv_heads < 256
        and seq_kv >= 2048
    )
    waves = 2 if split_kv else owner_waves
    threads = waves * 64
    waves_per_eu = (
        2
        if paired
        else _select_waves_per_eu(
            owner_waves=owner_waves,
            enough_prefill_parallelism=(
                not decode
                and batch_count * query_heads * (seq_q // 128)
                >= _FOUR_WAVE_PREFILL_MIN_CTAS
            ),
            seq_kv=seq_kv,
            qk_head_dim=qk_head_dim,
        )
    )
    pack_size = 8
    mma_tile_size = 32

    if (qk_head_dim, value_head_dim) not in ((128, 128), (192, 128)):
        raise ValueError(
            "FlyDSL forward requires (qk_head_dim, v_head_dim) "
            "to be (128, 128) or (192, 128)"
        )
    if sparse_query_size != 128 or sparse_kv_size != 128:
        raise ValueError("FlyDSL forward requires sparse block size 128")
    if seq_kv % sparse_kv_size:
        raise ValueError("FlyDSL forward requires Sk divisible by 128")
    if mask_batches not in (1, batch_count):
        raise ValueError("BlockMask batch dimension must be 1 or B")
    if mask_heads not in (1, kv_heads, query_heads):
        raise ValueError("BlockMask head dimension must be 1, Hkv, or Hq")
    if partial_capacity <= 0 or full_capacity <= 0:
        raise ValueError("FlyDSL forward requires non-empty index storage")
    if len(mask_buffer_shapes) != len(mask_buffer_strides):
        raise ValueError("mask buffer shape/stride descriptors must match")
    if len(mask_buffer_shapes) > 4:
        raise ValueError("FlyDSL forward supports at most four mask buffers")

    batch_count, query_heads, kv_heads, seq_q, seq_kv, qk_head_dim, value_head_dim = (
        map(
            int,
            (
                batch_count,
                query_heads,
                kv_heads,
                seq_q,
                seq_kv,
                qk_head_dim,
                value_head_dim,
            ),
        )
    )
    mask_batches, mask_heads, num_q_blocks, partial_capacity, full_capacity = map(
        int, (mask_batches, mask_heads, num_q_blocks, partial_capacity, full_capacity)
    )
    gqa_group = query_heads // kv_heads
    decode = bool(decode)
    pipelined = (not decode and seq_kv > sparse_kv_size) or (
        qk_head_dim == 128
        and decode
        and owner_waves in (2, 4)
        and batch_count * kv_heads <= 256
        and seq_kv >= 2048
    )
    packed_rows = gqa_group * seq_q if decode else seq_q
    query_chunks = (
        (packed_rows + query_rows - 1) // query_rows if decode else seq_q // query_rows
    )

    launch_heads = kv_heads if decode else query_heads
    total_heads = batch_count * launch_heads
    # Limit the heads interleaved across query blocks to preserve K/V reuse.
    heads_per_group = max(1, min(total_heads, 64))
    while total_heads % heads_per_group:
        heads_per_group -= 1

    if decode:
        if mask_heads not in (1, kv_heads):
            raise ValueError(
                "FlyDSL forward decode requires a shared or per-KV-head BlockMask"
            )
        if num_q_blocks != 1:
            raise ValueError("FlyDSL forward decode requires one sparse Q block")
        if packed_rows <= 0 or packed_rows > 256:
            raise ValueError("FlyDSL forward decode requires 1 <= (Hq/Hkv)*Sq <= 256")
    else:
        if seq_q % query_rows:
            raise ValueError(
                "FlyDSL forward prefill requires Sq divisible by its owner tile"
            )
        if num_q_blocks != seq_q // sparse_query_size:
            raise ValueError("BlockMask Q rows must cover Sq with 128-row blocks")

    tiles_per_sparse_block = sparse_kv_size // kv_tile_rows
    qk_reduction_steps = qk_head_dim // 16
    first_softmax_count = min(24, 2 * qk_reduction_steps)
    output_chunks = value_head_dim // mma_tile_size
    qk_packs_per_row = qk_head_dim // pack_size
    value_packs_per_row = value_head_dim // pack_size
    query_loads = (query_rows * qk_packs_per_row) // threads
    kv_load_threads = 64 if split_kv else threads
    key_loads = (kv_tile_rows * qk_packs_per_row) // kv_load_threads
    value_loads = (kv_tile_rows * value_packs_per_row) // kv_load_threads
    if (query_rows * qk_packs_per_row) % threads:
        raise ValueError("FlyDSL forward Q staging must evenly cover its tile")
    if (kv_tile_rows * qk_packs_per_row) % kv_load_threads:
        raise ValueError("FlyDSL forward K staging must evenly cover its tile")
    if (kv_tile_rows * value_packs_per_row) % kv_load_threads:
        raise ValueError("FlyDSL forward V staging must evenly cover its tile")

    def contiguous_stride(heads, sequence, dimension):
        return (heads * sequence * dimension, sequence * dimension, dimension, 1)

    q_stride = tuple(q_stride or contiguous_stride(query_heads, seq_q, qk_head_dim))
    k_stride = tuple(k_stride or contiguous_stride(kv_heads, seq_kv, qk_head_dim))
    v_stride = tuple(v_stride or contiguous_stride(kv_heads, seq_kv, value_head_dim))
    o_stride = tuple(o_stride or contiguous_stride(query_heads, seq_q, value_head_dim))
    scale_log2 = float(scale) * _LOG2E
    stats_in_log2 = bool(stats_in_log2)
    mask_program = tuple(mask_program)
    mask_output_slot = int(mask_output_slot)
    mask_buffer_shapes = tuple(tuple(shape) for shape in mask_buffer_shapes)
    mask_buffer_strides = tuple(tuple(stride) for stride in mask_buffer_strides)
    mask_buffer_count = len(mask_buffer_shapes)
    mask_buffer_sizes = tuple(
        1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
        for shape, strides in zip(mask_buffer_shapes, mask_buffer_strides)
    )
    mask_load_width = 4
    vector_mask_loads, supports_mask_intervals, flat_work_mask = analyze_mask_access(
        mask_program,
        mask_output_slot,
        mask_buffer_shapes,
        mask_buffer_strides,
        paired=paired,
        mask_load_width=mask_load_width,
    )

    ForwardSharedMemory = make_forward_shared_memory(
        pipelined,
        split_kv,
        kv_tile_rows,
        qk_head_dim,
        value_head_dim,
        query_rows,
        waves,
        output_chunks,
    )

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def kernel(
        query,
        key,
        value,
        logsumexp_input,
        max_scores_input,
        partial_counts_input,
        partial_indices_input,
        full_counts_input,
        full_indices_input,
        mask_buffer_0,
        mask_buffer_1,
        mask_buffer_2,
        mask_buffer_3,
        output,
    ):
        thread_id = fx.thread_idx.x
        # This gfx950-only kernel uses wave64, including on older FlyDSL releases.
        wave_size = 64
        wave_id = thread_id // i32(wave_size)
        wave = i32(fx.rocdl.readfirstlane(i32.ir_type, wave_id.ir_value()))
        if const_expr(paired):
            llvm.intr_assume(
                ((wave >= i32(0)) & (wave < i32(waves))).ir_value(), [], []
            )
        shared_memory = fx.SharedAllocator().allocate(ForwardSharedMemory).peek()

        def run_body(stagger):
            if const_expr(paired):
                lane_value = fx.thread_idx.x % i32(wave_size)
                lane_raw = llvm.inline_asm(
                    lane_value.ir_value().type,
                    [lane_value.ir_value()],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
                lane = i32(lane_raw)
                llvm.intr_assume(
                    ((lane >= i32(0)) & (lane < i32(wave_size))).ir_value(), [], []
                )
                thread_id = wave * i32(wave_size) + lane
            else:
                thread_id = fx.thread_idx.x
                lane = thread_id % i32(wave_size)

            lane_half = lane // i32(mma_tile_size)
            mma_atom = fx.make_mma_atom(
                fx.rocdl.MFMA(mma_tile_size, mma_tile_size, 16, bf16)
            )
            tiled_mma = fx.make_tiled_mma(mma_atom, layout((1, 1, 1), (1, 1, 1)))
            thread_mma = tiled_mma.get_slice(lane)
            accumulator_coords = thread_mma.partition_C(
                tensor_view(0, layout((mma_tile_size, mma_tile_size), (1, 0)))
            )
            query_coords = thread_mma.partition_C(
                tensor_view(0, layout((mma_tile_size, mma_tile_size), (0, 1)))
            )
            query_k_coords = thread_mma.partition_B(
                tensor_view(0, layout((mma_tile_size, qk_head_dim), (0, 1)))
            )
            if const_expr(paired):
                batch = fx.block_idx.z
            else:
                grouped_head = fx.block_idx.z * i32(heads_per_group) + fx.block_idx.x
                batch = (
                    i32(0)
                    if const_expr(batch_count == 1)
                    else grouped_head // i32(launch_heads)
                )
            if const_expr(decode or flat_work_mask):
                query_chunk = fx.block_idx.y
            else:
                # Dispatch longer causal query blocks first to avoid a ragged tail.
                query_chunk = i32(query_chunks - 1) - fx.block_idx.y
            query_base = query_chunk * i32(query_rows)
            if const_expr(paired):
                head = fx.block_idx.x
                if const_expr(gqa_group > 1):
                    head = (head % i32(kv_heads)) * i32(gqa_group) + head // i32(
                        kv_heads
                    )
                kv_head = head // i32(gqa_group)
            elif const_expr(decode):
                kv_head = grouped_head % i32(kv_heads)
                head = kv_head * i32(gqa_group)
            else:
                head = grouped_head % i32(query_heads)
                kv_head = head // i32(gqa_group)

            def row_coordinates(local_row):
                packed_row = query_base + local_row
                if const_expr(decode):
                    valid = packed_row < i32(packed_rows)
                    safe_row = valid.select(packed_row, i32(0))
                    row_head = kv_head * i32(gqa_group) + safe_row // i32(seq_q)
                    query_position = safe_row % i32(seq_q)
                else:
                    valid = i32(0) == i32(0)
                    row_head = head
                    query_position = packed_row
                return valid, row_head, query_position

            row_in_wave = i32(scalar(query_coords[0]))
            query_row = (
                row_in_wave
                if const_expr(split_kv)
                else wave * i32(mma_tile_size) + row_in_wave
            )
            query_valid, query_head, query_pos = row_coordinates(query_row)

            if const_expr(pipelined):
                # Q aliases K0/K1 until every wave has cached its fragments.
                query_shared_ptr = shared_memory.key_stage_0.ptr
                key_stages = [
                    shared_memory.key_stage_0.ptr,
                    shared_memory.key_stage_1.ptr,
                ]
                value_stages = [
                    shared_memory.value_stage_0.ptr,
                    shared_memory.value_stage_1.ptr,
                ]
            else:
                query_shared_ptr = shared_memory.query.ptr
                kv_shared_ptr = shared_memory.key_value.ptr
                if const_expr(split_kv):
                    kv_shared_ptr = fx.get_iter(
                        slice_view(
                            tensor_view(
                                kv_shared_ptr,
                                layout(
                                    (waves, kv_tile_rows * qk_head_dim),
                                    (kv_tile_rows * qk_head_dim, 1),
                                ),
                            ),
                            (wave, None),
                        )
                    )
                key_stages = [kv_shared_ptr]
                value_stages = [kv_shared_ptr]

            batch_i64 = i64(batch)
            kv_head_i64 = i64(kv_head)
            key_view = make_global_view(
                key,
                (batch_i64, kv_head_i64, None, None),
                (batch_count, kv_heads, seq_kv, qk_head_dim),
                k_stride,
            )
            value_view = make_global_view(
                value,
                (batch_i64, kv_head_i64, None, None),
                (batch_count, kv_heads, seq_kv, value_head_dim),
                v_stride,
            )

            def make_query_view(tensor, dimension, strides):
                if const_expr(decode):
                    return make_global_view(
                        tensor,
                        (batch_i64, None, kv_head_i64, None, None),
                        (batch_count, gqa_group, kv_heads, seq_q, dimension),
                        (
                            strides[0],
                            strides[1],
                            gqa_group * strides[1],
                            strides[2],
                            strides[3],
                        ),
                    )
                return make_global_view(
                    tensor,
                    (batch_i64, i64(head), None, None),
                    (batch_count, query_heads, seq_q, dimension),
                    strides,
                )

            query_view = make_query_view(query, qk_head_dim, q_stride)
            output_view = make_query_view(output, value_head_dim, o_stride)

            mask_rows = mask_batches * mask_heads * num_q_blocks

            def make_metadata_view(tensor, size):
                if const_expr(paired):
                    return tensor_view(fx.get_iter(tensor), layout(size, 1))
                return make_global_view(tensor, None, size, 1)

            partial_counts = make_metadata_view(partial_counts_input, mask_rows)
            partial_indices = make_metadata_view(
                partial_indices_input, mask_rows * partial_capacity
            )
            full_counts = make_metadata_view(full_counts_input, mask_rows)
            full_indices = make_metadata_view(
                full_indices_input, mask_rows * full_capacity
            )
            logsumexp_view = make_global_view(
                logsumexp_input, None, batch_count * query_heads * seq_q, 1
            )
            max_scores_view = make_global_view(
                max_scores_input, None, batch_count * query_heads * seq_q, 1
            )
            mask_buffers = make_mask_buffers(
                make_global_view,
                mask_buffer_count,
                mask_buffer_sizes,
                mask_buffer_0,
                mask_buffer_1,
                mask_buffer_2,
                mask_buffer_3,
            )

            output_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), bf16)

            def load_i32(view, index):
                return i32(view[index])

            def load_uniform(view, index):
                if const_expr(paired):
                    loaded = load_i32(view, index)
                    return i32(fx.rocdl.readfirstlane(i32.ir_type, loaded.ir_value()))
                return fx.gpu.shuffle_idx(load_i32(view, index), 0, wave_size)

            evaluate_mask = make_mask_evaluator(
                mask_program,
                mask_output_slot,
                mask_buffer_strides,
                mask_buffers,
                load_i32,
                batch,
                query_head,
            )

            mask_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), i32)

            def load_mask_group(first_key):
                inputs = [i32(batch), query_head, query_pos, first_key]
                groups = {}
                for slot, buffer_index, indices in vector_mask_loads:
                    offset = i32(0)
                    for dimension, index in enumerate(indices):
                        offset = offset + inputs[index] * i32(
                            mask_buffer_strides[buffer_index][dimension]
                        )
                    source = divide(
                        mask_buffers[buffer_index], layout(mask_load_width, 1)
                    )
                    fragment = fx.make_rmem_tensor(mask_load_width, i32)
                    tensor_copy(
                        mask_copy,
                        slice_view(source, (None, offset // i32(mask_load_width))),
                        fragment,
                    )
                    groups[slot] = Vec(fragment.load())
                return groups

            global_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), bf16)
            lds_copy = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), bf16)
            transposed_lds_copy = fx.make_copy_atom(
                fx.rocdl.cdna4.LDSReadTrans(16, 64), bf16
            )
            key_layout = make_qk_shared_layout(kv_tile_rows, qk_head_dim)
            value_layout = make_value_shared_layout(kv_tile_rows, value_head_dim)
            shared_keys = [tensor_view(pointer, key_layout) for pointer in key_stages]
            key_copy_coords = fx.make_composed_layout(
                fx.right_inverse(key_layout.outer),
                fx.make_composed_layout(
                    key_layout.inner, layout(kv_tile_rows * qk_head_dim, 1)
                ),
            )
            value_copy_coords = fx.right_inverse(value_layout)
            key_destinations = [
                divide(
                    tensor_view(pointer, layout(kv_tile_rows * qk_head_dim, 1)),
                    layout(pack_size, 1),
                )
                for pointer in key_stages
            ]
            value_destinations = [
                divide(
                    tensor_view(pointer, layout(kv_tile_rows * value_head_dim, 1)),
                    layout(pack_size, 1),
                )
                for pointer in value_stages
            ]
            if const_expr(paired):
                shared_query = query_view
            elif const_expr(not pipelined or not decode):
                shared_query = tensor_view(
                    query_shared_ptr, make_qk_shared_layout(query_rows, qk_head_dim)
                )
            else:
                # Decode inputs are contiguous; pack the group's query rows.
                shared_query = tensor_view(
                    fx.get_iter(query_view),
                    layout((packed_rows, qk_head_dim), (qk_head_dim, 1)),
                )
            query_wave = i32(0) if const_expr(split_kv) else wave
            query_tiles = fx.flat_divide(shared_query, (mma_tile_size, 16))
            key_tiles = [
                fx.flat_divide(shared_key, (mma_tile_size, 16))
                for shared_key in shared_keys
            ]
            query_copy = fx.make_tiled_copy_B(global_copy, tiled_mma).get_slice(lane)
            key_copy = fx.make_tiled_copy_A(global_copy, tiled_mma).get_slice(lane)
            value_copy = fx.make_tiled_copy_A(transposed_lds_copy, tiled_mma).get_slice(
                lane
            )
            shared_copy = fx.make_copy_atom(fx.UniversalCopy128b(), bf16)
            probability_coords = thread_mma.partition_B(
                tensor_view(0, layout((mma_tile_size, 16), (1, mma_tile_size)))
            )
            # Softmax keeps each lane's C values in register order. Permute V's
            # reduction mode to match that order without a cross-lane P shuffle.
            value_mma_layout = fx.composition(
                fx.select(value_layout, [1, 0]),
                fx.make_tile(
                    layout(value_head_dim, 1),
                    layout((4, 2, 2, kv_tile_rows // 16), (1, 8, 4, 16)),
                ),
            )
            value_tiles = [
                fx.flat_divide(
                    tensor_view(pointer, value_mma_layout), (mma_tile_size, 16)
                )
                for pointer in value_stages
            ]

            def mfma(a_fragment, b_fragment, accumulator_values):
                c_fragment = fx.make_fragment_like(accumulator_coords, f32)
                c_fragment.store(Vec(accumulator_values))
                fx.gemm(tiled_mma, c_fragment, a_fragment, b_fragment, c_fragment)
                return c_fragment.load()

            def get_mask_row():
                if const_expr(mask_batches == 1):
                    mask_batch = i32(0)
                else:
                    mask_batch = batch
                if const_expr(mask_heads == 1):
                    mask_head = i32(0)
                elif const_expr(mask_heads == kv_heads):
                    mask_head = kv_head
                else:
                    mask_head = head
                if const_expr(decode):
                    mask_q_block = i32(0)
                else:
                    mask_q_block = query_base // i32(sparse_query_size)
                mask_row = (mask_batch * i32(mask_heads) + mask_head) * i32(
                    num_q_blocks
                ) + mask_q_block
                return mask_row

            if const_expr(paired):
                mask_row = get_mask_row()
                full_count = load_uniform(full_counts, mask_row)
                partial_count = load_uniform(partial_counts, mask_row)
                partial_base = mask_row * i32(partial_capacity)
                second_full_count = load_uniform(full_counts, mask_row + i32(1))
                second_partial_count = load_uniform(partial_counts, mask_row + i32(1))
                pair_word, pair_cache, read_pair_cache, merge_rows = (
                    make_paired_metadata_ops(
                        lane,
                        load_i32,
                        wave_size,
                        seq_kv,
                        sparse_kv_size,
                    )
                )
                full_count, partial_count, pair_full_cache, partial_bits = merge_rows(
                    mask_row,
                    full_count,
                    second_full_count,
                    partial_count,
                    second_partial_count,
                    full_indices,
                    partial_indices,
                    full_capacity,
                    partial_capacity,
                )

            # Cache scaled Q once so its LDS reads do not compete with K/V staging.
            query_scale = vector([_f32(scale_log2)], f32).broadcast_to(pack_size)
            query_packs = []
            if const_expr(pipelined and not decode and not paired):
                query_layout = make_qk_shared_layout(query_rows, qk_head_dim)
                query_copy_coords = fx.make_composed_layout(
                    fx.right_inverse(query_layout.outer),
                    fx.make_composed_layout(
                        query_layout.inner, layout(query_rows * qk_head_dim, 1)
                    ),
                )
                query_destinations = divide(
                    tensor_view(query_shared_ptr, layout(query_rows * qk_head_dim, 1)),
                    layout(pack_size, 1),
                )
                for load_step in unroll(query_loads):
                    linear = i32(load_step * threads) + thread_id
                    logical = i32(
                        scalar(fx.crd2idx(linear * i32(pack_size), query_copy_coords))
                    )
                    row = logical % i32(query_rows)
                    chunk = logical // i32(query_rows * pack_size)
                    source = divide(
                        slice_view(query_view, (query_base + row, None)),
                        layout(pack_size, 1),
                    )
                    tensor_copy(
                        lds_copy,
                        slice_view(source, (None, chunk)),
                        slice_view(query_destinations, (None, linear)),
                    )
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.rocdl.s_barrier()
                for k_step in unroll(qk_reduction_steps):
                    tile = slice_view(query_tiles, (None, None, query_wave, k_step))
                    fragment = thread_mma.make_fragment_B(tile)
                    tensor_copy(
                        shared_copy,
                        query_copy.partition_S(tile),
                        query_copy.retile(fragment),
                    )
                    query_packs.append(
                        (Vec(fragment.load()).to(f32) * query_scale).to(bf16)
                    )
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.rocdl.s_barrier()
            elif const_expr(pipelined):
                if const_expr(paired):
                    query_source = slice_view(query_view, (query_pos, None))
                else:
                    local_head = query_head - kv_head * i32(gqa_group)
                    query_source = slice_view(query_view, (local_head, query_pos, None))
                query_row_packs = divide(query_source, layout(pack_size, 1))
                raw_query_packs = []
                for k_step in unroll(qk_reduction_steps):
                    column = i32(scalar(query_k_coords[0, 0, k_step]))
                    q_fragment = fx.make_rmem_tensor(pack_size, bf16)
                    tensor_copy(
                        global_copy,
                        slice_view(query_row_packs, (None, column // i32(pack_size))),
                        q_fragment,
                    )
                    raw_query_packs.append(Vec(q_fragment.load()))
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                for k_step in unroll(qk_reduction_steps):
                    query_packs.append(
                        (Vec(raw_query_packs[k_step].to(f32)) * query_scale).to(bf16)
                    )
            else:
                for load_step in unroll(query_loads):
                    linear = i32(load_step * threads) + thread_id
                    row = linear // i32(qk_packs_per_row)
                    chunk = linear % i32(qk_packs_per_row)
                    column = chunk * i32(pack_size)
                    row_valid, row_head, row_query_pos = row_coordinates(row)
                    query_fragment = fx.make_rmem_tensor(pack_size, bf16)
                    if const_expr(decode):
                        local_head = row_head - kv_head * i32(gqa_group)
                        source_row = slice_view(
                            query_view, (local_head, row_query_pos, None)
                        )
                    else:
                        source_row = slice_view(query_view, (row_query_pos, None))
                    source = divide(source_row, layout(pack_size, 1))
                    tensor_copy(
                        global_copy, slice_view(source, (None, chunk)), query_fragment
                    )
                    query_value = Vec(query_fragment.load())
                    if const_expr(decode):
                        query_value = vector(
                            [
                                row_valid.select(query_value[element], bf16(0.0))
                                for element in unroll(pack_size)
                            ],
                            bf16,
                        )
                    scaled_query = Vec(query_value.to(f32)) * query_scale
                    query_destination = divide(
                        slice_view(shared_query, (row, None)), layout(pack_size, 1)
                    )
                    query_fragment.store(Vec(scaled_query).to(bf16))
                    tensor_copy(
                        shared_copy,
                        query_fragment,
                        slice_view(query_destination, (None, chunk)),
                    )
                fx.gpu.barrier()

            def load_query(k_step):
                tile = slice_view(query_tiles, (None, None, query_wave, k_step))
                fragment = thread_mma.make_fragment_B(tile)
                if const_expr(pipelined):
                    fragment.store(Vec(query_packs[k_step]))
                else:
                    tensor_copy(
                        shared_copy,
                        query_copy.partition_S(tile),
                        query_copy.retile(fragment),
                    )
                return fragment

            def reduce_lane_pair(value, maximum):
                raw = i32(vector([value], f32).bitcast(i32)[0]).ir_value()
                swapped = fx.rocdl.permlane32_swap(
                    ir.Type.parse("!llvm.struct<(i32, i32)>"), raw, raw, False, True
                )
                pair = [
                    _f32(
                        vector(
                            [
                                i32(
                                    llvm.extractvalue(
                                        i32.ir_type, swapped, [element_index]
                                    )
                                )
                            ],
                            i32,
                        ).bitcast(f32)[0]
                    )
                    for element_index in unroll(2)
                ]
                lhs, rhs = pair
                return _maximum(lhs, rhs) if maximum else lhs + rhs

            zero16 = Vec.filled(16, 0.0, f32)
            output_accumulators = [zero16 for _ in unroll(output_chunks)]
            running_max = _f32(_NEG_BIG)
            running_sum = _f32(0.0)

            if const_expr(not paired):
                mask_row = get_mask_row()

            stage_key, stage_value, load_key, load_value = make_kv_staging(
                (key_copy, lds_copy, transposed_lds_copy, shared_copy, value_copy),
                (
                    key_copy_coords,
                    key_destinations,
                    key_stages,
                    k_stride,
                    key_tiles,
                    key_view,
                ),
                (
                    value_copy_coords,
                    value_destinations,
                    value_stages,
                    v_stride,
                    value_tiles,
                    value_view,
                ),
                (
                    key_loads,
                    kv_load_threads,
                    kv_tile_rows,
                    pack_size,
                    qk_head_dim,
                    value_head_dim,
                    value_loads,
                    wave_size,
                ),
                (lane, paired, split_kv, thread_mma, thread_id, wave),
            )

            def accumulate_probability(pack_values, pack_index, tile_output, stage):
                probability_fragment = fx.make_fragment_like(probability_coords, bf16)
                probability_fragment.store(Vec(pack_values))
                for d_chunk in unroll(output_chunks):
                    value_pack = load_value(pack_index, d_chunk, stage)
                    tile_output[d_chunk] = mfma(
                        value_pack, probability_fragment, tile_output[d_chunk]
                    )
                return tile_output

            process_tile = make_tile_processor(
                (
                    key_loads,
                    kv_tile_rows,
                    mma_tile_size,
                    output_chunks,
                    qk_reduction_steps,
                    value_loads,
                    wave_size,
                ),
                (
                    accumulator_coords,
                    batch,
                    query_base,
                    query_head,
                    query_pos,
                    query_valid,
                    wave,
                ),
                (
                    evaluate_mask,
                    supports_mask_intervals,
                    load_i32,
                    load_mask_group,
                    mask_buffers,
                    mask_buffer_count,
                    mask_output_slot,
                    mask_program,
                    mask_buffer_strides,
                    mask_load_width,
                    vector_mask_loads,
                ),
                (
                    accumulate_probability,
                    load_key,
                    load_query,
                    mfma,
                    reduce_lane_pair,
                    stage_key,
                    stage_value,
                    zero16,
                ),
                (paired, pipelined),
            )

            def process_pipelined_run(
                block_count,
                block_indices,
                block_base,
                masked,
                run_state,
                index_cache=None,
            ):
                run_results = run_state
                if block_count > i32(0):
                    if const_expr(index_cache is None):
                        first_block = load_uniform(block_indices, block_base)
                    else:
                        first_block = read_pair_cache(index_cache, i32(0))
                    first_chunk = first_block * i32(tiles_per_sparse_block)
                    stage_key(first_chunk * i32(kv_tile_rows), 0)
                    stage_value(first_chunk * i32(kv_tile_rows), 0)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    if const_expr(paired):
                        phase_fence()
                    else:
                        fx.gpu.barrier()

                    pipeline_state = [first_block, *run_state]
                    pipeline_results = pipeline_state
                    for block_index, iter_args in range(
                        i32(0), block_count, i32(1), init=pipeline_state
                    ):
                        current_block = i32(iter_args[0])
                        iter_max = _f32(iter_args[1])
                        iter_sum = _f32(iter_args[2])
                        iteration_exact_max = _f32(iter_args[3 + output_chunks])
                        iter_output = [
                            iter_args[3 + d_chunk] for d_chunk in unroll(output_chunks)
                        ]

                        if const_expr(paired):
                            # A union block may be absent from this query block's row.
                            own_full = u32(0)
                            own_partial = u32(0)
                            word_index = current_block // i32(32)
                            for word in unroll(len(owner_full_bits)):
                                selected = word_index == i32(word)
                                own_full = own_full | selected.select(
                                    owner_full_bits[word], u32(0)
                                )
                                own_partial = own_partial | selected.select(
                                    owner_partial_bits[word], u32(0)
                                )
                            bit = u32(1) << (u32(current_block) & u32(31))
                            full = (own_full & bit) != u32(0)
                            active = ((own_full | own_partial) & bit) != u32(0)
                        else:
                            active = None
                            full = None

                        first_chunk = current_block * i32(tiles_per_sparse_block)
                        second_chunk = first_chunk + i32(1)
                        stage_key(second_chunk * i32(kv_tile_rows), 1)
                        stage_value(second_chunk * i32(kv_tile_rows), 1)
                        iter_output, iter_max, iter_sum, iteration_exact_max = (
                            process_tile(
                                first_chunk,
                                masked,
                                iter_output,
                                iter_max,
                                iter_sum,
                                stage=0,
                                exact_max=iteration_exact_max,
                                active=active,
                                full=full,
                            )
                        )
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        if const_expr(paired):
                            phase_fence()
                        else:
                            fx.gpu.barrier()

                        next_index = i32(block_index) + i32(1)
                        next_block = current_block
                        if next_index < block_count:
                            if const_expr(index_cache is None):
                                next_block = load_uniform(
                                    block_indices, block_base + next_index
                                )
                            else:
                                next_block = read_pair_cache(index_cache, next_index)
                            next_chunk = next_block * i32(tiles_per_sparse_block)
                            stage_key(next_chunk * i32(kv_tile_rows), 0)
                            stage_value(next_chunk * i32(kv_tile_rows), 0)
                        iter_output, iter_max, iter_sum, iteration_exact_max = (
                            process_tile(
                                second_chunk,
                                masked,
                                iter_output,
                                iter_max,
                                iter_sum,
                                stage=1,
                                exact_max=iteration_exact_max,
                                active=active,
                                full=full,
                            )
                        )
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        if const_expr(paired):
                            phase_fence()
                        else:
                            fx.gpu.barrier()

                        pipeline_results = yield [
                            next_block,
                            iter_max,
                            iter_sum,
                            *iter_output,
                            iteration_exact_max,
                        ]
                    run_results = pipeline_results[1:]
                return run_results

            def process_sparse_run(
                block_count, block_indices, block_base, masked, run_state
            ):
                run_results = run_state
                for block_index, iter_args in range(
                    i32(0), block_count, i32(1), init=run_state
                ):
                    iter_max = _f32(iter_args[0])
                    iter_sum = _f32(iter_args[1])
                    iter_output = [
                        iter_args[2 + d_chunk] for d_chunk in unroll(output_chunks)
                    ]
                    sparse_block = load_uniform(
                        block_indices, block_base + i32(block_index)
                    )
                    for sub_block in unroll(tiles_per_sparse_block):
                        iter_output, iter_max, iter_sum, ignored_exact_max = (
                            process_tile(
                                sparse_block * i32(tiles_per_sparse_block)
                                + i32(sub_block),
                                masked,
                                iter_output,
                                iter_max,
                                iter_sum,
                            )
                        )
                    run_results = yield [iter_max, iter_sum] + iter_output

                return run_results

            def process_split_run(
                block_count, block_indices, block_base, masked, run_state
            ):
                run_results = run_state
                split_count = (block_count + i32(waves - 1)) // i32(waves)
                for split_index, iter_args in range(
                    i32(0), split_count, i32(1), init=run_state
                ):
                    iter_max = _f32(iter_args[0])
                    iter_sum = _f32(iter_args[1])
                    iter_output = [
                        iter_args[2 + d_chunk] for d_chunk in unroll(output_chunks)
                    ]
                    block_index = i32(split_index * waves) + wave
                    active = block_index < block_count
                    safe_index = active.select(block_index, i32(0))
                    sparse_block = load_uniform(block_indices, block_base + safe_index)
                    for sub_block in unroll(tiles_per_sparse_block):
                        iter_output, iter_max, iter_sum, ignored_exact_max = (
                            process_tile(
                                sparse_block * i32(tiles_per_sparse_block)
                                + i32(sub_block),
                                masked,
                                iter_output,
                                iter_max,
                                iter_sum,
                                active=active,
                            )
                        )
                    run_results = yield [iter_max, iter_sum] + iter_output
                return run_results

            (
                staged_phase,
                phase_exp,
                phase_pack_sum,
                load_phase_value,
                phase_pv,
                phase_shift,
                pin_output_chunks,
                phase_fence,
                load_phase_key,
                phase_qk,
                phase_max,
            ) = make_staged_pipeline(
                (
                    key_loads,
                    kv_tile_rows,
                    output_chunks,
                    qk_head_dim,
                    qk_reduction_steps,
                    first_softmax_count,
                    value_head_dim,
                    value_loads,
                ),
                (
                    load_key,
                    load_query,
                    load_value,
                    mfma,
                    probability_coords,
                    reduce_lane_pair,
                    stage_key,
                    stage_value,
                    zero16,
                ),
                stagger,
            )

            def process_staged_full(block_count, run_state):
                results = run_state
                if block_count > i32(0):
                    index_cache = pair_full_cache
                    first_block = read_pair_cache(index_cache, i32(0))
                    first_chunk = first_block * i32(2)
                    stage_key(first_chunk * i32(kv_tile_rows), 0)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    stage_key((first_chunk + i32(1)) * i32(kv_tile_rows), 1)
                    stage_value(first_chunk * i32(kv_tile_rows), 0)
                    if const_expr(stagger):
                        phase_fence()
                    key_packs = load_phase_key(0, 0)
                    fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    score_lo = phase_qk(key_packs)
                    phase_fence()
                    key_packs = load_phase_key(1, 0)
                    fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    score_hi = phase_qk(key_packs)
                    peak = phase_max(score_lo, score_hi)
                    exact_peak = _maximum(_f32(run_state[2 + output_chunks]), peak)
                    correction = _exp2(_f32(run_state[0]) - peak)
                    correction_vec = vector([correction], f32).broadcast_to(16)
                    tile_output = [
                        Vec(run_state[2 + output_chunk]) * correction_vec
                        for output_chunk in unroll(output_chunks)
                    ]
                    tile_sum = _f32(run_state[1]) * correction
                    pending = phase_exp(phase_shift(score_lo, score_hi, peak), 0, 16)
                    fx.rocdl.s_waitcnt(vmcnt=value_loads, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    next_slot = (block_count > i32(1)).select(i32(1), i32(0))
                    next_block = read_pair_cache(index_cache, next_slot)
                    stage_key(next_block * i32(sparse_kv_size), 0)
                    state = [
                        first_block,
                        next_block,
                        peak,
                        tile_sum,
                        exact_peak,
                        *tile_output,
                        *pending,
                    ]
                    pipeline_results = state
                    for block_index, args in range(
                        i32(0), block_count, i32(1), init=state
                    ):
                        current_block = i32(args[0])
                        maximum = _f32(args[2])
                        total = _f32(args[3])
                        exact = _f32(args[4])
                        outputs = [
                            args[5 + output_chunk]
                            for output_chunk in unroll(output_chunks)
                        ]
                        shifted = [
                            _f32(args[5 + output_chunks + element_index])
                            for element_index in unroll(32)
                        ]
                        next_index = i32(block_index) + i32(1)
                        following_block = i32(args[1])
                        lookahead_block = following_block
                        shifted, outputs, maximum, total, exact = staged_phase(
                            current_block * i32(2) + i32(1),
                            following_block * i32(2) + i32(1),
                            1,
                            shifted,
                            outputs,
                            maximum,
                            total,
                            exact,
                        )
                        if next_index < block_count:
                            lookahead = next_index + i32(1)
                            lookahead_safe = (lookahead < block_count).select(
                                lookahead, block_count - i32(1)
                            )
                            lookahead_block = read_pair_cache(
                                index_cache, lookahead_safe
                            )
                            shifted, outputs, maximum, total, exact = staged_phase(
                                following_block * i32(2),
                                lookahead_block * i32(2),
                                0,
                                shifted,
                                outputs,
                                maximum,
                                total,
                                exact,
                            )
                        pipeline_results = yield [
                            following_block,
                            lookahead_block,
                            maximum,
                            total,
                            exact,
                            *outputs,
                            *shifted,
                        ]
                    fx.rocdl.s_waitcnt(vmcnt=key_loads, lgkmcnt=0, expcnt=0)
                    phase_fence()
                    maximum = _f32(pipeline_results[2])
                    total = _f32(pipeline_results[3])
                    exact = _f32(pipeline_results[4])
                    outputs = [
                        pipeline_results[5 + output_chunk]
                        for output_chunk in unroll(output_chunks)
                    ]
                    shifted = [
                        _f32(pipeline_results[5 + output_chunks + element_index])
                        for element_index in unroll(32)
                    ]
                    probabilities, probability_sum = phase_pack_sum(
                        phase_exp(shifted, 16, 32)
                    )
                    total = total + probability_sum
                    for half in unroll(2):
                        value_packs = load_phase_value(half, 1)
                        fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
                        phase_fence()
                        outputs = phase_pv(value_packs, probabilities, outputs, half)
                        phase_fence()
                    if const_expr(not stagger):
                        phase_fence()
                    results = [maximum, total, *outputs, exact]
                return results

            def store_results(final_results, logsumexp, max_scores_input):
                final_max = _f32(final_results[0])
                final_sum = _f32(final_results[1])
                output_values = [
                    final_results[2 + d_chunk] for d_chunk in unroll(output_chunks)
                ]

                inverse_sum = (final_sum > _f32(0.0)).select(
                    _f32(fx.rocdl.rcp(f32.ir_type, final_sum.ir_value())), _f32(0.0)
                )
                inverse_vector = vector([inverse_sum], f32).broadcast_to(16)

                if const_expr(decode):
                    local_head = query_head - kv_head * i32(gqa_group)
                    output_source = slice_view(
                        output_view, (local_head, query_pos, None)
                    )
                else:
                    output_source = slice_view(output_view, (query_pos, None))
                output_row = divide(output_source, layout(8, 1))
                store_valid = query_valid
                if const_expr(split_kv):
                    store_valid = store_valid & (wave == i32(0))
                if store_valid:
                    _store_output_fragments(
                        output_values,
                        inverse_vector,
                        output_row,
                        lane_half,
                        output_copy,
                        output_chunks,
                        mma_tile_size,
                    )

                if store_valid & (lane_half == i32(0)):
                    has_values = final_sum > _f32(0.0)
                    lse_value = final_max + fx.math.log2(final_sum)
                    max_value = (
                        _f32(final_results[2 + output_chunks])
                        if const_expr(pipelined)
                        else final_max
                    )
                    if const_expr(not stats_in_log2):
                        lse_value = lse_value * _f32(_LN2)
                        max_value = max_value * _f32(_LN2)
                    lse_value = has_values.select(lse_value, _f32(float("-inf")))
                    max_value = has_values.select(max_value, _f32(float("-inf")))
                    stats_offset = (batch * i32(query_heads) + query_head) * i32(
                        seq_q
                    ) + query_pos
                    logsumexp[stats_offset] = lse_value
                    if const_expr(write_max_scores):
                        max_scores_input[stats_offset] = max_value

            if const_expr(paired):
                initial_state = [running_max, running_sum] + output_accumulators

                full_results = process_staged_full(
                    full_count, [*initial_state, running_max]
                )
                # Build owner membership after the full-block pipeline to limit liveness.
                owner_row = mask_row + i32(1 if stagger else 0)
                owner_full_count = load_uniform(full_counts, owner_row)
                owner_partial_count = load_uniform(partial_counts, owner_row)
                owner_full_bits = [
                    pair_word(
                        full_indices, owner_row, owner_full_count, full_capacity, word
                    )
                    for word in unroll(len(partial_bits))
                ]
                owner_partial_bits = [
                    pair_word(
                        partial_indices,
                        owner_row,
                        owner_partial_count,
                        partial_capacity,
                        word,
                    )
                    for word in unroll(len(partial_bits))
                ]
                partial_cache = pair_cache(partial_bits)
                final_results = process_pipelined_run(
                    partial_count,
                    partial_indices,
                    partial_base,
                    True,
                    full_results,
                    partial_cache,
                )
                store_results(final_results, logsumexp_view, max_scores_view)
            else:
                full_count = load_uniform(full_counts, mask_row)
                partial_count = load_uniform(partial_counts, mask_row)
                full_base = mask_row * i32(full_capacity)
                partial_base = mask_row * i32(partial_capacity)
                initial_state = [running_max, running_sum] + output_accumulators

                if const_expr(split_kv):
                    full_results = process_split_run(
                        full_count, full_indices, full_base, False, initial_state
                    )
                    split_results = process_split_run(
                        partial_count, partial_indices, partial_base, True, full_results
                    )
                    final_results = reduce_split_results(
                        split_results,
                        shared_memory,
                        lane_half,
                        wave,
                        row_in_wave,
                        output_chunks,
                    )
                elif const_expr(pipelined):
                    full_results = process_pipelined_run(
                        full_count,
                        full_indices,
                        full_base,
                        False,
                        [*initial_state, running_max],
                    )
                    final_results = process_pipelined_run(
                        partial_count, partial_indices, partial_base, True, full_results
                    )
                else:
                    full_results = process_sparse_run(
                        full_count, full_indices, full_base, False, initial_state
                    )
                    final_results = process_sparse_run(
                        partial_count, partial_indices, partial_base, True, full_results
                    )
                store_results(final_results, logsumexp_view, max_scores_view)

        if const_expr(paired):
            fx.rocdl.sched_barrier(0)
            if wave >= i32(owner_waves // 2):
                run_body(True)
            fx.rocdl.sched_barrier(0)
            if wave < i32(owner_waves // 2):
                run_body(False)
        else:
            run_body(False)

    return _make_forward_launch(
        kernel,
        grid=(
            (query_heads, query_chunks, batch_count)
            if paired
            else (heads_per_group, query_chunks, total_heads // heads_per_group)
        ),
        threads=threads,
        mask_buffer_count=mask_buffer_count,
        waves_per_eu=waves_per_eu,
    )
