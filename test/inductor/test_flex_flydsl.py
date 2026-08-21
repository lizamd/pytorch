# Owner(s): ["module: inductor"]
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl.flydsl_template import FlyDSLTemplate
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch._inductor.kernel.flex import flex_flydsl_attention
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    _can_use_flydsl_flex_attention_backward,
    flex_flydsl_backward_template,
    flex_flydsl_forward_template,
)
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _score_graph(fn) -> SimpleNamespace:
    gm = torch.fx.symbolic_trace(fn)
    return SimpleNamespace(graph_module=gm)


def _identity_score_graph() -> SimpleNamespace:
    return _score_graph(lambda score, b, h, m, n: score)


def _nontrivial_score_graph() -> SimpleNamespace:
    return _score_graph(lambda score, b, h, m, n: score * 2.0)


def _fake_query(dtype: torch.dtype) -> SimpleNamespace:
    return SimpleNamespace(get_dtype=lambda: dtype)


class TestFlexFlyDSLGates(TestCase):
    def test_template_registered(self):
        self.assertIn("flex_flydsl_forward", FlyDSLTemplate.all_templates)
        self.assertIs(
            FlyDSLTemplate.all_templates["flex_flydsl_forward"],
            flex_flydsl_forward_template,
        )
        self.assertIn("flex_flydsl_backward", FlyDSLTemplate.all_templates)
        self.assertIs(
            FlyDSLTemplate.all_templates["flex_flydsl_backward"],
            flex_flydsl_backward_template,
        )

    def test_gate_declines_when_runtime_unavailable(self):
        with mock.patch.object(
            flex_flydsl_attention, "runtime_available", return_value=False
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                _identity_score_graph(),
                _identity_score_graph(),
                _fake_query(torch.bfloat16),
            )
        self.assertFalse(can_use)
        self.assertIn("unavailable", reason)

    def test_gate_declines_when_not_rocm(self):
        with (
            mock.patch.object(
                flex_flydsl_attention, "runtime_available", return_value=True
            ),
            mock.patch.object(torch.version, "hip", None),
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                _identity_score_graph(),
                _identity_score_graph(),
                _fake_query(torch.bfloat16),
            )
        self.assertFalse(can_use)
        self.assertIn("ROCm", reason)

    def test_gate_declines_when_not_bf16(self):
        with (
            mock.patch.object(
                flex_flydsl_attention, "runtime_available", return_value=True
            ),
            mock.patch.object(torch.version, "hip", "6.0.0"),
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                _identity_score_graph(),
                _identity_score_graph(),
                _fake_query(torch.float16),
            )
        self.assertFalse(can_use)
        self.assertIn("bf16", reason)

    def test_gate_declines_when_score_mod_nontrivial(self):
        with (
            mock.patch.object(
                flex_flydsl_attention, "runtime_available", return_value=True
            ),
            mock.patch.object(torch.version, "hip", "6.0.0"),
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                _nontrivial_score_graph(),
                _identity_score_graph(),
                _fake_query(torch.bfloat16),
            )
        self.assertFalse(can_use)
        self.assertIn("identity score_mod", reason)


@unittest.skipUnless(runtime_available(), "flydsl runtime not available")
class TestFlexFlyDSLRuntime(TestCase):
    def _compare_backward(
        self,
        q,
        k,
        v,
        grad_out,
        *,
        block_mask=None,
        scale=None,
        atol=0.1,
        rtol=0.05,
    ):
        def clone_preserve_strides(tensor):
            clone = torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            clone.copy_(tensor)
            return clone.requires_grad_(True)

        def compile_and_backward(backend):
            q_arg = clone_preserve_strides(q)
            k_arg = clone_preserve_strides(k)
            v_arg = clone_preserve_strides(v)
            torch._dynamo.reset()
            compiled = torch.compile(
                lambda q_in, k_in, v_in: flex_attention(
                    q_in,
                    k_in,
                    v_in,
                    block_mask=block_mask,
                    scale=scale,
                    kernel_options={"BACKEND": backend},
                ),
                fullgraph=True,
            )
            out = compiled(q_arg, k_arg, v_arg)
            out.backward(grad_out)
            torch.cuda.synchronize()
            return out, q_arg.grad, k_arg.grad, v_arg.grad

        flydsl = compile_and_backward("FLYDSL")
        triton = compile_and_backward("TRITON")
        for name, actual, expected in zip(
            ("output", "dQ", "dK", "dV"),
            flydsl,
            triton,
            strict=True,
        ):
            torch.testing.assert_close(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                msg=f"{name} mismatch",
            )
        return flydsl, triton

    def test_gate_allows_trivial_bf16_on_rocm(self):
        can_use, reason = _can_use_flydsl_flex_attention_backward(
            _identity_score_graph(),
            _identity_score_graph(),
            _fake_query(torch.bfloat16),
        )
        self.assertTrue(can_use, reason)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_causal_mha_backward_matches_triton(self):
        batch, heads, seq, head_dim = 1, 2, 256, 128
        scale = 0.07
        torch.manual_seed(0)

        kv_num_blocks = torch.ones(1, 1, 2, device="cuda", dtype=torch.int32)
        kv_indices = torch.tensor(
            [[[[0], [1]]]],
            device="cuda",
            dtype=torch.int32,
        )
        full_kv_num_blocks = torch.tensor(
            [[[0, 1]]],
            device="cuda",
            dtype=torch.int32,
        )
        full_kv_indices = torch.tensor(
            [[[[0], [0]]]],
            device="cuda",
            dtype=torch.int32,
        )

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            mask_mod=causal,
            seq_lengths=(seq, seq),
        )

        q = torch.randn(
            batch,
            heads,
            seq,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        grad_out = torch.randn_like(q)

        def run(backend, q_arg, k_arg, v_arg):
            return flex_attention(
                q_arg,
                k_arg,
                v_arg,
                block_mask=block_mask,
                scale=scale,
                kernel_options={"BACKEND": backend},
            )

        def compile_and_backward(backend):
            q_arg = q.detach().clone().requires_grad_(True)
            k_arg = k.detach().clone().requires_grad_(True)
            v_arg = v.detach().clone().requires_grad_(True)
            torch._dynamo.reset()
            compiled = torch.compile(
                lambda q_in, k_in, v_in: run(backend, q_in, k_in, v_in),
                fullgraph=True,
            )
            out = compiled(q_arg, k_arg, v_arg)
            out.backward(grad_out)
            torch.cuda.synchronize()
            return out, q_arg.grad, k_arg.grad, v_arg.grad

        flydsl = compile_and_backward("FLYDSL")
        triton = compile_and_backward("TRITON")

        for actual, expected in zip(flydsl, triton, strict=True):
            torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.05)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_dense_mha_backward_matches_triton(self):
        batch, heads, seq, head_dim = 1, 1, 256, 128
        torch.manual_seed(1)

        kv_num_blocks = torch.zeros(1, 1, 2, device="cuda", dtype=torch.int32)
        kv_indices = torch.zeros(1, 1, 2, 1, device="cuda", dtype=torch.int32)
        full_kv_num_blocks = torch.full((1, 1, 2), 2, device="cuda", dtype=torch.int32)
        full_kv_indices = torch.tensor(
            [[[[0, 1], [0, 1]]]],
            device="cuda",
            dtype=torch.int32,
        )
        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            seq_lengths=(seq, seq),
        )

        q = torch.randn(
            batch,
            heads,
            seq,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        grad_out = torch.randn_like(q)

        def compile_and_backward(backend):
            q_arg = q.detach().clone().requires_grad_(True)
            k_arg = k.detach().clone().requires_grad_(True)
            v_arg = v.detach().clone().requires_grad_(True)
            torch._dynamo.reset()
            compiled = torch.compile(
                lambda q_in, k_in, v_in: flex_attention(
                    q_in,
                    k_in,
                    v_in,
                    block_mask=block_mask,
                    kernel_options={"BACKEND": backend},
                ),
                fullgraph=True,
            )
            out = compiled(q_arg, k_arg, v_arg)
            out.backward(grad_out)
            torch.cuda.synchronize()
            return out, q_arg.grad, k_arg.grad, v_arg.grad

        flydsl = compile_and_backward("FLYDSL")
        triton = compile_and_backward("TRITON")

        for actual, expected in zip(flydsl, triton, strict=True):
            torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.05)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_qk192_v128_dense_and_causal_backward_matches_triton(self):
        batch, heads, seq = 1, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        scale = 0.07
        torch.manual_seed(2)
        q = torch.randn(
            batch,
            heads,
            seq,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch,
            heads,
            seq,
            v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad_out = torch.randn_like(v)

        def dense(b, h, q_idx, kv_idx):
            del b, h, kv_idx
            return q_idx >= 0

        dense_block_mask = create_block_mask(
            dense,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=dense_block_mask,
            scale=scale,
        )

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=block_mask,
            scale=scale,
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_qk192_v128_sliding_window_backward_matches_triton(self):
        batch, heads, seq = 1, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        torch.manual_seed(3)
        q = torch.randn(
            batch,
            heads,
            seq,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch,
            heads,
            seq,
            v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad_out = torch.randn_like(v)

        def sliding_window(b, h, q_idx, kv_idx):
            del b, h
            return (q_idx >= kv_idx) & (q_idx - kv_idx < 96)

        block_mask = create_block_mask(
            sliding_window,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=block_mask,
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_qk192_v128_per_head_mask_backward_matches_triton(self):
        batch, heads, seq = 1, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        torch.manual_seed(31)
        q = torch.randn(
            batch,
            heads,
            seq,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch,
            heads,
            seq,
            v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad_out = torch.randn_like(v)

        def per_head_window(b, h, q_idx, kv_idx):
            del b
            return (q_idx >= kv_idx) & (q_idx - kv_idx < 64 + h * 64)

        block_mask = create_block_mask(
            per_head_window,
            1,
            heads,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=block_mask,
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_qk192_v128_padded_positive_strides_backward_matches_triton(self):
        batch, heads, seq = 1, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        torch.manual_seed(32)

        def random_strided(size, stride):
            tensor = torch.empty_strided(
                size,
                stride,
                device="cuda",
                dtype=torch.bfloat16,
            )
            tensor.normal_()
            return tensor

        q_head_stride = seq * 208 + 64
        k_head_stride = seq * 216 + 32
        v_head_stride = seq * 136 + 64
        do_head_stride = seq * 144 + 32
        q = random_strided(
            (batch, heads, seq, qk_head_dim),
            (heads * q_head_stride + 128, q_head_stride, 208, 1),
        )
        k = random_strided(
            (batch, heads, seq, qk_head_dim),
            (heads * k_head_stride + 64, k_head_stride, 216, 1),
        )
        v = random_strided(
            (batch, heads, seq, v_head_dim),
            (heads * v_head_stride + 128, v_head_stride, 136, 1),
        )
        grad_out = random_strided(
            (batch, heads, seq, v_head_dim),
            (heads * do_head_stride + 64, do_head_stride, 144, 1),
        )

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=block_mask,
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_qk192_v128_transposed_document_backward_matches_triton(self):
        batch, heads, seq = 2, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        scale = 0.07
        torch.manual_seed(4)
        q = torch.randn(
            batch,
            seq,
            heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).transpose(1, 2)
        k = torch.randn_like(q)
        v = torch.randn(
            batch,
            seq,
            heads,
            v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).transpose(1, 2)
        grad_out = torch.randn_like(v)
        document_end = torch.tensor(
            [
                [127] * 128 + [255] * 128,
                [63] * 64 + [191] * 128 + [255] * 64,
            ],
            device="cuda",
            dtype=torch.int32,
        )

        def document_causal(b, h, q_idx, kv_idx):
            del h
            return (q_idx >= kv_idx) & (q_idx <= document_end[b, kv_idx])

        block_mask = create_block_mask(
            document_causal,
            batch,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        (output, dq, dk, dv), _ = self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=block_mask,
            scale=scale,
        )
        self.assertEqual(output.stride()[-1], 1)
        self.assertEqual(dq.stride(), q.stride())
        self.assertEqual(dk.stride(), k.stride())
        self.assertEqual(dv.stride(), v.stride())

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_qk192_v128_b1_document_two_captures_backward_matches_triton(self):
        batch, heads, seq = 1, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        torch.manual_seed(5)
        q = torch.randn(
            batch,
            heads,
            seq,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch,
            heads,
            seq,
            v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad_out = torch.randn_like(v)
        document_ids = torch.arange(seq, device="cuda", dtype=torch.int32) // 128
        document_starts = torch.tensor(
            [0, 128],
            device="cuda",
            dtype=torch.int32,
        )

        def document_causal(b, h, q_idx, kv_idx):
            del b, h
            return (kv_idx >= document_starts[document_ids[q_idx]]) & (
                kv_idx <= q_idx
            )

        block_mask = create_block_mask(
            document_causal,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=block_mask,
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_qk192_v128_standard_width_document_backward_matches_triton(self):
        batch, heads, seq = 1, 1, 4096
        qk_head_dim, v_head_dim = 192, 128
        scale = 0.07
        torch.manual_seed(6)
        q = torch.randn(
            batch,
            heads,
            seq,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch,
            heads,
            seq,
            v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad_out = torch.randn_like(v)
        document_size = 512
        document_end = (
            torch.arange(seq, device="cuda", dtype=torch.int32) // document_size + 1
        ) * document_size - 1

        def document_causal(b, h, q_idx, kv_idx):
            del b, h
            return (q_idx >= kv_idx) & (q_idx <= document_end[kv_idx])

        block_mask = create_block_mask(
            document_causal,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self.assertEqual(block_mask.kv_indices.shape[-1], seq // 128)
        self.assertEqual(block_mask.full_kv_indices.shape[-1], seq // 128)
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=block_mask,
            scale=scale,
            atol=0.12,
            rtol=0.06,
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950",
        "requires ROCm gfx950",
    )
    def test_forward_only_shapes_are_rejected_by_backward_gate(self):
        batch, seq = 1, 256
        kv_num_blocks = torch.zeros(1, 1, 2, device="cuda", dtype=torch.int32)
        kv_indices = torch.zeros(1, 1, 2, 1, device="cuda", dtype=torch.int32)
        full_kv_num_blocks = torch.full((1, 1, 2), 2, device="cuda", dtype=torch.int32)
        full_kv_indices = torch.tensor(
            [[[[0, 1], [0, 1]]]],
            device="cuda",
            dtype=torch.int32,
        )
        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            seq_lengths=(seq, seq),
        )

        cases = (("gqa", 4, 2, 128, 128),)
        for name, q_heads, kv_heads, qk_dim, v_dim in cases:
            with self.subTest(name=name):
                q = torch.randn(
                    batch,
                    q_heads,
                    seq,
                    qk_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                k = torch.randn(
                    batch,
                    kv_heads,
                    seq,
                    qk_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                v = torch.randn(
                    batch,
                    kv_heads,
                    seq,
                    v_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                grad_out = torch.randn(
                    batch,
                    q_heads,
                    seq,
                    v_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )

                torch._dynamo.reset()
                compiled = torch.compile(
                    lambda q_in, k_in, v_in: flex_attention(
                        q_in,
                        k_in,
                        v_in,
                        block_mask=block_mask,
                        enable_gqa=q_heads != kv_heads,
                        kernel_options={"BACKEND": "FLYDSL"},
                    ),
                    fullgraph=True,
                )
                with self.assertRaisesRegex(RuntimeError, "MHA with matching Q/K"):
                    out = compiled(q, k, v)
                    out.backward(grad_out)


if __name__ == "__main__":
    run_tests()
