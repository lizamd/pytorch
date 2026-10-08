# mypy: allow-untyped-defs

from typing import Any

import torch

from ...codegen.flydsl import flydsl_utils
from ...virtualized import V
from .common import construct_strides, infer_dense_strides
from .flex_flash_attention import is_trivial_mask_graph, is_trivial_score_graph
from .flex_flydsl_mask import lower_flydsl_mask_graph


_MAX_BUFFER_BYTES = 1 << 32


def _contiguous_strides(shape):
    return construct_strides(shape, range(len(shape) - 1, -1, -1))


def _get_supported_bhsd_stride(node, *, allow_strided: bool) -> tuple[int, ...] | None:
    try:
        sizes = [V.graph.sizevars.guard_int(value) for value in node.get_size()]
        strides = [V.graph.sizevars.guard_int(value) for value in node.get_stride()]
    except (TypeError, ValueError):
        return None
    if len(sizes) != 4 or len(strides) != 4:
        return None
    if strides == _contiguous_strides(sizes):
        return tuple(strides)
    if allow_strided and strides[-1] == 1 and all(stride > 0 for stride in strides):
        return tuple(strides)
    return None


def _fits_u32_head_slice(node) -> bool:
    try:
        sizes = [V.graph.sizevars.guard_int(value) for value in node.get_size()]
        strides = [V.graph.sizevars.guard_int(value) for value in node.get_stride()]
        element_size = torch._utils._element_size(node.get_dtype())
    except (AttributeError, TypeError, ValueError):
        return False
    if len(sizes) != 4 or len(strides) != 4 or any(stride < 0 for stride in strides):
        return False
    storage_elements = 1 + sum(
        (size - 1) * stride for size, stride in zip(sizes[-2:], strides[-2:])
    )
    return storage_elements * element_size < _MAX_BUFFER_BYTES


def _is_contiguous_shape_stride(
    shape: tuple[int, ...], stride: tuple[int, ...]
) -> bool:
    if len(shape) != len(stride):
        return False
    expected = _contiguous_strides(tuple(max(size, 1) for size in shape))
    return all(
        size == 1 or actual == contiguous
        for size, actual, contiguous in zip(shape, stride, expected)
    )


def _is_gfx950_device(device) -> bool:
    if not flydsl_utils.runtime_available() or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")
    return str(arch).split(":", 1)[0] == "gfx950"


def _check_flydsl_common_compatibility(
    *,
    query,
    key,
    value,
    subgraph,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    allow_mask_mod_buffers: bool = False,
    allow_strided_bhsd: bool = False,
) -> str:
    device = query.get_device()
    if device is None or device.type != "cuda" or not _is_gfx950_device(device):
        return "requires ROCm gfx950 and the FlyDSL runtime"
    if query.get_dtype() != torch.bfloat16:
        return f"supports BF16 only, got {query.get_dtype()}"
    if query.get_dtype() != key.get_dtype() or query.get_dtype() != value.get_dtype():
        return "requires query, key, and value to have the same dtype"
    if not is_trivial_score_graph(subgraph.graph_module):
        return "supports identity score_mod only"
    if score_mod_other_buffers:
        return "does not support captured score_mod buffers"
    if mask_mod_other_buffers and not allow_mask_mod_buffers:
        return "does not support captured mask_mod buffers"

    tensors = (query, key, value)
    if not all(
        _get_supported_bhsd_stride(node, allow_strided=allow_strided_bhsd) is not None
        for node in tensors
        if node is not None
    ):
        layout = "4D BHSD tensors with contiguous head dimensions"
        if not allow_strided_bhsd:
            layout = "contiguous 4D BHSD tensors"
        return f"requires {layout}"
    if not all(_fits_u32_head_slice(node) for node in tensors):
        return "requires every per-head tensor slice to be smaller than 4 GiB"
    return ""


def _get_flydsl_flex_attention_forward_config(
    *,
    query,
    key,
    value,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    subgraph,
    mask_graph,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    scale,
    sparse_q_block_size,
    sparse_kv_block_size,
) -> tuple[dict[str, Any] | None, str]:
    if full_kv_num_blocks is None or full_kv_indices is None:
        return None, "requires full_kv_num_blocks/full_kv_indices metadata"

    metadata_nodes = (kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices)
    try:
        batch_size, query_heads, query_length, qk_head_dim = [
            V.graph.sizevars.guard_int(item) for item in query.get_size()
        ]
        key_batch_size, kv_heads, kv_length, key_dim = [
            V.graph.sizevars.guard_int(item) for item in key.get_size()
        ]
        value_batch_size, value_heads, value_length, value_head_dim = [
            V.graph.sizevars.guard_int(item) for item in value.get_size()
        ]
        metadata_shapes = tuple(
            tuple(V.graph.sizevars.guard_int(item) for item in node.get_size())
            for node in metadata_nodes
        )
        mask_shape, index_shape, full_count_shape, full_index_shape = metadata_shapes
        metadata_dtypes = tuple(node.get_dtype() for node in metadata_nodes)
        metadata_devices = tuple(node.get_device() for node in metadata_nodes)
        sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
        sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
        full_numel = V.graph.sizevars.guard_int(full_kv_num_blocks.get_numel())
        scale_value = float(scale)
        output_stride = tuple(
            V.graph.sizevars.guard_int(item)
            for item in infer_dense_strides(
                [batch_size, query_heads, query_length, value_head_dim],
                query.get_stride(),
            )
        )
    except (AttributeError, TypeError, ValueError):
        return None, "requires statically known tensor and BlockMask dimensions"

    if any(dtype != torch.int32 for dtype in metadata_dtypes):
        return None, "requires int32 BlockMask metadata"
    if any(device != query.get_device() for device in metadata_devices):
        return None, "requires BlockMask metadata on the query device"
    try:
        metadata_strides = tuple(
            tuple(V.graph.sizevars.guard_int(item) for item in node.get_stride())
            for node in metadata_nodes
        )
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None, "requires statically known BlockMask metadata strides"
    if not all(
        _is_contiguous_shape_stride(shape, stride)
        for shape, stride in zip(metadata_shapes, metadata_strides)
    ):
        return None, "requires contiguous BlockMask metadata"

    trivial_mask = is_trivial_mask_graph(mask_graph.graph_module)
    mask_program = None
    if not trivial_mask:
        mask_program, mask_reason = lower_flydsl_mask_graph(
            mask_graph.graph_module, mask_mod_other_buffers
        )
        if mask_program is None:
            return None, f"unsupported mask_mod: {mask_reason}"

    decode = 0 < query_length < 128
    common_reason = _check_flydsl_common_compatibility(
        query=query,
        key=key,
        value=value,
        subgraph=subgraph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        allow_mask_mod_buffers=mask_program is not None,
        allow_strided_bhsd=not decode,
    )
    if common_reason:
        return None, common_reason

    q_stride = _get_supported_bhsd_stride(query, allow_strided=not decode)
    k_stride = _get_supported_bhsd_stride(key, allow_strided=not decode)
    v_stride = _get_supported_bhsd_stride(value, allow_strided=not decode)
    if q_stride is None or k_stride is None or v_stride is None:
        return None, "requires supported Q/K/V BHSD strides"

    if (key_batch_size, kv_heads, kv_length) != (
        value_batch_size,
        value_heads,
        value_length,
    ):
        return None, "requires key and value to have matching B/Hkv/Sk dimensions"
    if batch_size != key_batch_size:
        return None, "does not yet support broadcasted K/V batches"
    if qk_head_dim != key_dim:
        return None, "requires query and key to have the same head dimension"
    if (qk_head_dim, value_head_dim) not in ((128, 128), (192, 128)):
        return (
            None,
            "supports only (QK head dim, V head dim) = (128, 128) or (192, 128)",
        )
    if (
        kv_heads <= 0
        or query_heads % kv_heads != 0
        or query_length <= 0
        or kv_length <= 0
    ):
        return None, "requires positive lengths and Hq divisible by Hkv"
    if len(mask_shape) != 3 or len(index_shape) != 4:
        return None, "requires 3D BlockMask counts and 4D BlockMask indices"
    if index_shape[:3] != mask_shape:
        return None, "requires matching BlockMask count/index leading dimensions"
    if mask_shape[0] not in (1, batch_size):
        return None, "BlockMask batch dimension must be 1 or B"
    if mask_shape[1] not in (1, kv_heads, query_heads):
        return None, "BlockMask head dimension must be 1, Hkv, or Hq"
    if sparse_q_block_size <= 0 or sparse_kv_block_size <= 0:
        return None, "requires positive sparse block sizes"

    has_full_blocks = full_numel != 0
    max_full_blocks = 1
    if has_full_blocks:
        if full_count_shape != mask_shape:
            return None, "requires matching partial/full BlockMask count dimensions"
        if len(full_index_shape) != 4 or full_index_shape[:3] != mask_shape:
            return None, "requires matching full BlockMask count/index dimensions"
        max_full_blocks = full_index_shape[-1]

    gqa_group_size = query_heads // kv_heads
    packed_decode_rows = gqa_group_size * query_length
    supports_prefill = query_length % 128 == 0 and mask_shape[2] == query_length // 128
    supports_decode = (
        decode
        and mask_shape[1] in (1, kv_heads)
        and mask_shape[2] == 1
        and 0 < packed_decode_rows <= 256
    )
    if kv_length % 128 != 0:
        return None, "requires Sk divisible by 128"
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        return None, "requires sparse Q/KV block sizes of 128"
    if not has_full_blocks or max_full_blocks <= 0 or index_shape[-1] <= 0:
        return None, "requires non-empty partial and full BlockMask storage"
    if not (supports_prefill or supports_decode):
        return (
            None,
            "requires prefill Sq divisible by 128 with matching BlockMask rows, "
            "or decode 0 < Sq < 128 with a shared/per-KV-head BlockMask and "
            "(Hq/Hkv)*Sq <= 256",
        )

    return (
        {
            "BATCH_SIZE": batch_size,
            "NUM_Q_HEADS": query_heads,
            "NUM_KV_HEADS": kv_heads,
            "SEQ_Q": query_length,
            "SEQ_KV": kv_length,
            "QK_HEAD_DIM": qk_head_dim,
            "V_HEAD_DIM": value_head_dim,
            "BLOCK_MASK_BATCH": mask_shape[0],
            "BLOCK_MASK_HEADS": mask_shape[1],
            "NUM_Q_BLOCKS": mask_shape[2],
            "MAX_PARTIAL_BLOCKS": index_shape[-1],
            "MAX_FULL_BLOCKS": max_full_blocks,
            "SPARSE_Q_BLOCK_SIZE": sparse_q_block_size,
            "SPARSE_KV_BLOCK_SIZE": sparse_kv_block_size,
            "MASK_PROGRAM": mask_program.instructions if mask_program else (),
            "MASK_PROGRAM_OUTPUT": mask_program.output if mask_program else 0,
            "MASK_BUFFER_COUNT": mask_program.buffer_count if mask_program else 0,
            "MASK_BUFFER_SHAPES": mask_program.buffer_shapes if mask_program else (),
            "MASK_BUFFER_STRIDES": mask_program.buffer_strides if mask_program else (),
            "SM_SCALE": scale_value,
            "Q_STRIDE": q_stride,
            "K_STRIDE": k_stride,
            "V_STRIDE": v_stride,
            "O_STRIDE": output_stride,
        },
        "",
    )
