# mypy: allow-untyped-defs

from .flex_attn_fwd_helpers import (
    _exp2,
    _f32,
    _maximum,
    _pin,
    _reduce32,
    bf16,
    const_expr,
    f32,
    flyc,
    fx,
    i32,
    layout,
    schedule_group,
    slice_view,
    u64,
    unroll,
    Vec,
    vector,
)


@flyc.jit
def make_staged_pipeline(geometry, compute, stagger):
    (
        key_loads,
        kv_tile_rows,
        output_chunks,
        qk_head_dim,
        qk_reduction_steps,
        first_softmax_count,
        value_head_dim,
        value_loads,
    ) = geometry
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
    ) = compute

    def phase_fence():
        fx.rocdl.sched_barrier(0)
        fx.rocdl.s_barrier()
        fx.rocdl.sched_barrier(0)

    def pin_vector(values):
        return Vec(_pin(Vec(values)))

    def pin_output_chunks(values):
        elements = []
        for part in unroll(2):
            chunk = pin_vector(
                vector(
                    [
                        _f32(Vec(values)[part * 8 + element_index])
                        for element_index in unroll(8)
                    ],
                    f32,
                )
            )
            elements.extend([_f32(chunk[element_index]) for element_index in unroll(8)])
        return vector(elements, f32)

    def load_phase_key(half, stage):
        return [load_key(step, half, stage) for step in unroll(qk_reduction_steps)]

    def phase_qk(packs):
        scores = zero16
        for step in unroll(qk_reduction_steps):
            scores = mfma(packs[step], load_query(step), scores)
        return Vec(scores)

    def load_phase_value(half, stage):
        return [
            load_value(pack, output_chunk, stage)
            for output_chunk in unroll(half * 2, half * 2 + 2)
            for pack in unroll(4)
        ]

    def phase_pv(packs, probabilities, tile_output, half):
        for output_chunk in unroll(2):
            for pack in unroll(4):
                probability_fragment = fx.make_fragment_like(probability_coords, bf16)
                probability_fragment.store(Vec(probabilities[pack]))
                out_index = half * 2 + output_chunk
                tile_output[out_index] = mfma(
                    packs[output_chunk * 4 + pack],
                    probability_fragment,
                    tile_output[out_index],
                )
        return tile_output

    def phase_exp(pending, begin, end):
        values = list(pending)
        for element in unroll(begin, end):
            values[element] = _exp2(_f32(pending[element]))
        return values

    def phase_pack_sum(probabilities):
        local = _reduce32(probabilities, False)
        total = reduce_lane_pair(local, False)
        packs = [
            vector(probabilities[pack * 8 : pack * 8 + 8], f32).to(bf16)
            for pack in unroll(4)
        ]
        return packs, total

    def phase_max(score_lo, score_hi):
        levels = [_f32(score_lo[element_index]) for element_index in unroll(16)] + [
            _f32(score_hi[element_index]) for element_index in unroll(16)
        ]
        return reduce_lane_pair(_reduce32(levels, True), True)

    def phase_shift(score_lo, score_hi, maximum, begin=0):
        shifted = []
        for part in unroll(begin, 4):
            score = score_lo if part < 2 else score_hi
            offset = (part % 2) * 8
            values = vector(
                [
                    _f32(score[offset + element_index]) - maximum
                    for element_index in unroll(8)
                ],
                f32,
            )
            values = pin_vector(values)
            shifted.extend([_f32(values[element_index]) for element_index in unroll(8)])
        return shifted

    def finish_shift(score_lo, score_hi, maximum, leading):
        shifted = [_f32(leading[element_index]) for element_index in unroll(8)]
        shifted.extend(phase_shift(score_lo, score_hi, maximum, 1))
        shifted = phase_exp(shifted, 0, 16)
        materialized = pin_vector(vector(shifted[:16], f32))
        return [
            _f32(materialized[element_index]) for element_index in unroll(16)
        ] + shifted[16:]

    def update_maximum(score_lo, score_hi, peak, exact_peak):
        tile_max = phase_max(score_lo, score_hi)
        exact_peak = _maximum(exact_peak, tile_max)
        rescale = u64(
            fx.rocdl.ballot(u64.ir_type, (tile_max > peak + _f32(8.0)).ir_value())
        ) != u64(0)
        new_peak = rescale.select(_maximum(peak, tile_max), peak)
        leading = pin_vector(
            vector(
                [
                    _f32(score_lo[element_index]) - new_peak
                    for element_index in unroll(8)
                ],
                f32,
            )
        )
        return exact_peak, rescale, new_peak, leading

    def staged_phase(
        kv_chunk,
        prefetch_chunk,
        stage,
        pending,
        tile_output,
        peak,
        tile_sum,
        exact_peak,
    ):
        if const_expr(qk_head_dim <= value_head_dim):
            key_lo = load_phase_key(0, stage)
            key_hi = load_phase_key(1, stage)
            stage_value(kv_chunk * i32(kv_tile_rows), stage)
            fx.rocdl.s_waitcnt(vmcnt=key_loads + value_loads, lgkmcnt=0, expcnt=0)
            phase_fence()

            score_lo = zero16
            score_hi = zero16
            for step in unroll(qk_reduction_steps):
                query_fragment = load_query(step)
                score_lo = Vec(mfma(key_lo[step], query_fragment, score_lo))
                score_hi = Vec(mfma(key_hi[step], query_fragment, score_hi))
            previous = phase_exp(pending, 16, 32)
            partial_sum = vector(previous[:first_softmax_count], f32).reduce(
                "add", fastmath="reassoc"
            )
            final_sum = vector(previous[first_softmax_count:], f32).reduce(
                "add", fastmath="reassoc"
            )
            tile_sum = tile_sum + reduce_lane_pair(partial_sum + final_sum, False)
            tile_sum = _f32(_pin(tile_sum))
            pinned = pin_vector(vector(previous, f32))
            previous = [_f32(pinned[element_index]) for element_index in unroll(32)]
            probabilities = [
                pin_vector(vector(previous[part * 8 : part * 8 + 8], f32).to(bf16))
                for part in unroll(4)
            ]
            for step in unroll(2 * qk_reduction_steps):
                schedule_group(0x08, 1, 1)
                if const_expr(step < 8):
                    schedule_group(0x400, 2, 1)
                else:
                    schedule_group(0x02, 5, 1)
            phase_fence()

        else:
            key_packs = load_phase_key(0, stage)
            if const_expr(stagger):
                stage_value(kv_chunk * i32(kv_tile_rows), stage)
            fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
            phase_fence()

            score_lo = phase_qk(key_packs)
            previous_probabilities = phase_exp(pending, 16, 24)
            partial_sum = vector(
                previous_probabilities[:first_softmax_count], f32
            ).reduce("add", fastmath="reassoc")
            partial_sum = _f32(_pin(partial_sum))
            probabilities = []
            for part in unroll(first_softmax_count // 8):
                values = vector(
                    previous_probabilities[part * 8 : part * 8 + 8], f32
                ).to(bf16)
                probabilities.append(pin_vector(values))
            remaining = previous_probabilities[first_softmax_count:32]
            for step in unroll(qk_reduction_steps):
                schedule_group(0x08, 1, 1)
                if const_expr(step < 4):
                    schedule_group(0x400, 2, 1)
                else:
                    schedule_group(0x02, first_softmax_count // 8 + 1, 1)
            phase_fence()

            key_packs = load_phase_key(1, stage)
            if const_expr(not stagger):
                stage_value(kv_chunk * i32(kv_tile_rows), stage)
            fx.rocdl.s_waitcnt(vmcnt=key_loads + value_loads, lgkmcnt=0, expcnt=0)
            phase_fence()

            score_hi = phase_qk(key_packs)
            final_probabilities = remaining[: 24 - first_softmax_count] + [
                _exp2(_f32(remaining[24 - first_softmax_count + element_index]))
                for element_index in unroll(8)
            ]
            final_sum = vector(final_probabilities, f32).reduce(
                "add", fastmath="reassoc"
            )
            local_sum = partial_sum + final_sum
            probability_sum = reduce_lane_pair(local_sum, False)
            for part in unroll((32 - first_softmax_count) // 8):
                probabilities.append(
                    pin_vector(
                        vector(final_probabilities[part * 8 : part * 8 + 8], f32).to(
                            bf16
                        )
                    )
                )
            tile_sum = tile_sum + probability_sum
            for step in unroll(qk_reduction_steps):
                schedule_group(0x08, 1, 2)
                if const_expr(step < 4):
                    schedule_group(0x400, 2, 2)
                else:
                    schedule_group(0x02, (32 - first_softmax_count) // 4 + 1, 2)
            phase_fence()

        if const_expr(qk_head_dim <= value_head_dim):
            value_packs = load_phase_value(0, 1 - stage) + load_phase_value(
                1, 1 - stage
            )
            stage_key(prefetch_chunk * i32(kv_tile_rows), stage)
            fx.rocdl.s_waitcnt(vmcnt=key_loads + value_loads, lgkmcnt=0, expcnt=0)
            phase_fence()

            probability_fragment = fx.make_fragment_like(probability_coords, bf16)
            probability_fragment.store(Vec(probabilities[0]))
            for output_chunk in unroll(output_chunks):
                tile_output[output_chunk] = mfma(
                    value_packs[output_chunk * 4],
                    probability_fragment,
                    tile_output[output_chunk],
                )
            exact_peak, rescale, new_peak, leading = update_maximum(
                score_lo, score_hi, peak, exact_peak
            )
            for step in unroll(output_chunks):
                schedule_group(0x08, 1, 3)
                schedule_group(0x02, 6, 3)
            fx.rocdl.sched_barrier(0)

            for pack in unroll(1, 4):
                probability_fragment = fx.make_fragment_like(probability_coords, bf16)
                probability_fragment.store(Vec(probabilities[pack]))
                for output_chunk in unroll(output_chunks):
                    tile_output[output_chunk] = mfma(
                        value_packs[output_chunk * 4 + pack],
                        probability_fragment,
                        tile_output[output_chunk],
                    )
            shifted = finish_shift(score_lo, score_hi, new_peak, leading)
            for step in unroll(3 * output_chunks):
                schedule_group(0x08, 1, 4)
                if const_expr(step < 8):
                    schedule_group(0x02, 3, 4)
                if const_expr(step >= 4):
                    schedule_group(0x400, 2, 4)
        else:
            value_packs = load_phase_value(0, 1 - stage)
            if const_expr(stagger):
                stage_key(prefetch_chunk * i32(kv_tile_rows), stage, 0, key_loads // 3)
            else:
                stage_key(prefetch_chunk * i32(kv_tile_rows), stage)
            fx.rocdl.s_waitcnt(vmcnt=63, lgkmcnt=0, expcnt=0)
            phase_fence()

            tile_output = phase_pv(value_packs, probabilities, tile_output, 0)
            exact_peak, rescale, new_peak, leading = update_maximum(
                score_lo, score_hi, peak, exact_peak
            )
            for step in unroll(8):
                schedule_group(0x08, 1, 3)
                schedule_group(0x02, 4, 3)
            phase_fence()

            value_packs = load_phase_value(1, 1 - stage)
            if const_expr(stagger):
                stage_key(
                    prefetch_chunk * i32(kv_tile_rows), stage, key_loads // 3, key_loads
                )
            fx.rocdl.s_waitcnt(vmcnt=key_loads + value_loads, lgkmcnt=0, expcnt=0)
            phase_fence()

            tile_output = phase_pv(value_packs, probabilities, tile_output, 1)
            shifted = finish_shift(score_lo, score_hi, new_peak, leading)
            for step in unroll(8):
                schedule_group(0x08, 1, 4)
                schedule_group(0x02, 3, 4)
                schedule_group(0x400, 2, 4)
        tile_output = [
            pin_output_chunks(output_values) for output_values in tile_output
        ]
        if rescale:
            correction = _exp2(peak - new_peak)
            correction_vec = vector([correction], f32).broadcast_to(16)
            tile_output = [
                Vec(output_values) * correction_vec for output_values in tile_output
            ]
            tile_sum = tile_sum * correction
        tile_output = [
            pin_output_chunks(output_values) for output_values in tile_output
        ]
        phase_fence()
        return shifted, tile_output, new_peak, tile_sum, exact_peak

    return (
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
    )


@flyc.jit
def reduce_split_results(
    split_results,
    shared_memory,
    lane_half,
    wave,
    row_in_wave,
    output_chunks,
):
    split_max = _f32(split_results[0])
    split_sum = _f32(split_results[1])
    split_output = [split_results[2 + d_chunk] for d_chunk in unroll(output_chunks)]
    reduction = slice_view(
        shared_memory.reduction_stats.view(layout((2, 2), (2, 1))),
        (lane_half, None),
    )
    reduction_outputs = slice_view(
        shared_memory.reduction_output.view(
            layout((2, 16, output_chunks), (output_chunks * 16, 1, 16))
        ),
        (lane_half, None, None),
    )
    if (wave == i32(1)) & (row_in_wave == i32(0)):
        reduction[0] = split_max
        reduction[1] = split_sum
        for d_chunk in unroll(output_chunks):
            slice_view(reduction_outputs, (None, d_chunk)).store(
                Vec(split_output[d_chunk])
            )
    fx.gpu.barrier()

    other_max = _f32(reduction[0])
    other_sum = _f32(reduction[1])
    combined_max = _maximum(split_max, other_max)
    split_scale = _exp2(split_max - combined_max)
    other_scale = _exp2(other_max - combined_max)
    split_sum = split_sum * split_scale + other_sum * other_scale
    split_max = combined_max
    split_scale_vec = vector([split_scale], f32).broadcast_to(16)
    other_scale_vec = vector([other_scale], f32).broadcast_to(16)
    for d_chunk in unroll(output_chunks):
        other_output = Vec(slice_view(reduction_outputs, (None, d_chunk)).load())
        split_output[d_chunk] = (
            Vec(split_output[d_chunk]) * split_scale_vec
            + other_output * other_scale_vec
        )
    return [split_max, split_sum] + split_output
