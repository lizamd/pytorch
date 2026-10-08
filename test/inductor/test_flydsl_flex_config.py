# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.kernel.flex.flex_flydsl_config import (
    _get_flydsl_flex_attention_forward_config,
    _get_supported_bhsd_stride,
    _is_contiguous_shape_stride,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)


class _FakeNode:
    def __init__(self, size, stride, dtype=torch.bfloat16, numel=None):
        self._size = list(size)
        self._stride = list(stride)
        self._dtype = dtype
        self._numel = int(torch.tensor(size).prod().item()) if numel is None else numel

    def get_size(self):
        return self._size

    def get_stride(self):
        return self._stride

    def get_dtype(self):
        return self._dtype

    def get_device(self):
        return torch.device("cuda", 0)

    def get_numel(self):
        return self._numel


def _contiguous_stride(size):
    stride = [1] * len(size)
    for index in range(len(size) - 2, -1, -1):
        stride[index] = stride[index + 1] * size[index + 1]
    return stride


def _fake_graph():
    return SimpleNamespace(
        sizevars=SimpleNamespace(guard_int=lambda value: int(value), shape_env=None)
    )


def _supported_fake_forward_inputs(
    *,
    query_shape=(1, 64, 512, 128),
    key_shape=(1, 4, 1024, 128),
    value_shape=None,
    mask_heads=1,
    width=8,
    dtype=torch.bfloat16,
    mask_fn=lambda batch_index, head_index, query, kv_index: query + 512 >= kv_index,
):
    value_shape = key_shape if value_shape is None else value_shape
    count_size = [1, mask_heads, max(1, query_shape[2] // 128)]
    sizes = (
        query_shape,
        key_shape,
        value_shape,
        query_shape[:3],
        query_shape[:3],
        count_size,
        [*count_size, width],
        count_size,
        [*count_size, width],
    )
    names = (
        "query",
        "key",
        "value",
        "logsumexp",
        "max_scores",
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
    )
    dtypes = (
        dtype,
        dtype,
        dtype,
        torch.float32,
        torch.float32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
    )
    inputs = {
        name: _FakeNode(size, _contiguous_stride(size), dtype)
        for name, size, dtype in zip(names, sizes, dtypes)
    }
    inputs.update(
        subgraph=SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(
                lambda score, batch_index, head_index, query, kv_index: score
            )
        ),
        mask_graph=SimpleNamespace(graph_module=torch.fx.symbolic_trace(mask_fn)),
        score_mod_other_buffers=[],
        mask_mod_other_buffers=[],
        scale=query_shape[-1] ** -0.5,
        sparse_q_block_size=128,
        sparse_kv_block_size=128,
    )
    return inputs


def _fake_config_result(inputs):
    config_names = (
        "query",
        "key",
        "value",
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "subgraph",
        "mask_graph",
        "score_mod_other_buffers",
        "mask_mod_other_buffers",
        "scale",
        "sparse_q_block_size",
        "sparse_kv_block_size",
    )
    with (
        V.set_graph_handler(_fake_graph()),
        mock.patch(
            "torch._inductor.kernel.flex.flex_flydsl_config._is_gfx950_device",
            return_value=True,
        ),
    ):
        return _get_flydsl_flex_attention_forward_config(
            **{name: inputs[name] for name in config_names}
        )


@instantiate_parametrized_tests
class TestFlyDSLFlexLayout(TestCase):
    @parametrize(
        "shape,stride,strict,metadata",
        [
            subtest(((2, 3, 4, 5), (60, 20, 5, 1), True, True), name="contiguous"),
            subtest(
                ((2, 1, 4, 5), (20, 99, 5, 1), False, True), name="size_one_dimension"
            ),
            subtest(((2, 3, 4, 5), (120, 40, 5, 1), False, False), name="strided"),
            subtest(
                ((2, 0, 4, 5), (0, 20, 5, 1), True, False), name="zero_size_strict"
            ),
            subtest(
                ((2, 0, 4, 5), (20, 20, 5, 1), False, True), name="zero_size_metadata"
            ),
            subtest(((2, 3, 4, 5), (20, 5, 1), False, False), name="rank_mismatch"),
            subtest(((), (), False, True), name="scalar_metadata"),
        ],
    )
    def test_contiguous_stride_checks(self, shape, stride, strict, metadata):
        graph = SimpleNamespace(sizevars=SimpleNamespace(guard_int=int))
        node = SimpleNamespace(get_size=lambda: shape, get_stride=lambda: stride)
        with V.set_graph_handler(graph):
            self.assertEqual(
                _get_supported_bhsd_stride(node, allow_strided=False),
                stride if strict else None,
            )
            self.assertEqual(_is_contiguous_shape_stride(shape, stride), metadata)


@instantiate_parametrized_tests
class TestFlyDSLFlexAttentionLowering(TestCase):
    @parametrize(
        "batch,query_heads,kv_heads,query_length,kv_length,qk_head_dim,value_head_dim,mask_heads,width,offset",
        [
            subtest((1, 64, 4, 512, 1024, 128, 128, 1, 8, 512), name="bf16_gqa"),
            subtest((1, 16, 16, 256, 256, 192, 128, 1, 1, 0), name="qk192_v128"),
            subtest((32, 64, 4, 4, 8192, 128, 128, 4, 16, 8188), name="gqa_decode_q4"),
        ],
    )
    def test_supported_forward_configs(
        self,
        batch,
        query_heads,
        kv_heads,
        query_length,
        kv_length,
        qk_head_dim,
        value_head_dim,
        mask_heads,
        width,
        offset,
    ):
        inputs = _supported_fake_forward_inputs(
            query_shape=(batch, query_heads, query_length, qk_head_dim),
            key_shape=(batch, kv_heads, kv_length, qk_head_dim),
            value_shape=(batch, kv_heads, kv_length, value_head_dim),
            mask_heads=mask_heads,
            width=width,
            mask_fn=lambda batch_index, head_index, query, kv_index: query + offset
            >= kv_index,
        )
        kwargs, reason = _fake_config_result(inputs)
        self.assertIsNotNone(kwargs, reason)
        keys = (
            "NUM_Q_HEADS",
            "NUM_KV_HEADS",
            "SEQ_Q",
            "SEQ_KV",
            "QK_HEAD_DIM",
            "V_HEAD_DIM",
            "BLOCK_MASK_HEADS",
        )
        self.assertEqual(
            tuple(kwargs[key] for key in keys),
            (
                query_heads,
                kv_heads,
                query_length,
                kv_length,
                qk_head_dim,
                value_head_dim,
                mask_heads,
            ),
        )
        self.assertNotIn("CAUSAL_PARTIAL_BLOCKS", kwargs)
        self.assertTrue(kwargs["MASK_PROGRAM"])
        self.assertEqual(kwargs["SPARSE_Q_BLOCK_SIZE"], 128)
        self.assertEqual(kwargs["SPARSE_KV_BLOCK_SIZE"], 128)

    @parametrize(
        "query_shape,key_shape,mask_heads,width,case",
        [
            ((1, 64, 512, 128), (1, 4, 1024, 128), 1, 8, "count_dtype"),
            ((1, 8, 256, 128), (1, 2, 256, 128), 1, 2, "q_block_size"),
            ((32, 64, 4, 128), (32, 4, 8192, 128), 64, 16, "decode_per_q_head_mask"),
            ((1, 16, 1, 128), (1, 1, 1 << 24, 128), 1, 16, "four_gib_head_slice"),
            ((1, 8, 256, 128), (1, 8, 256, 128), 1, 8, "dtype"),
        ],
    )
    def test_unsupported_choices_fall_back(
        self, query_shape, key_shape, mask_heads, width, case
    ):
        inputs = _supported_fake_forward_inputs(
            query_shape=query_shape,
            key_shape=key_shape,
            mask_heads=mask_heads,
            width=width,
            dtype=torch.float16 if case == "dtype" else torch.bfloat16,
            mask_fn=lambda batch_index, head_index, query, kv_index: query
            + key_shape[2]
            - query_shape[2]
            >= kv_index,
        )
        if case == "count_dtype":
            inputs["kv_num_blocks"] = _FakeNode([1, 1, 4], [4, 4, 1], torch.float32)
        elif case == "q_block_size":
            inputs["sparse_q_block_size"] = 256
        reasons = {
            "count_dtype": "requires int32 BlockMask metadata",
            "q_block_size": "requires sparse Q/KV block sizes of 128",
            "decode_per_q_head_mask": "requires prefill Sq divisible by 128",
            "four_gib_head_slice": "requires every per-head tensor slice to be smaller than 4 GiB",
            "dtype": "supports BF16 only",
        }
        kwargs, reason = _fake_config_result(inputs)
        self.assertIsNone(kwargs)
        self.assertIn(reasons[case], reason)

    def test_four_gib_kv_buffer_uses_rebased_head_slice(self):
        inputs = _supported_fake_forward_inputs(
            query_shape=(64, 64, 4, 128),
            key_shape=(64, 4, 65536, 128),
            mask_heads=4,
            width=16,
            mask_fn=lambda batch_index, head_index, query, kv_index: query + 65532
            >= kv_index,
        )
        inputs["key"]._numel = 1 << 31
        kwargs, reason = _fake_config_result(inputs)
        self.assertIsNotNone(kwargs, reason)


if __name__ == "__main__":
    run_tests()
