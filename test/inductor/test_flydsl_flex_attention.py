# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

from test_flydsl_flex_config import (
    _fake_graph,
    _FakeNode,
    _supported_fake_forward_inputs,
)

import torch
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    flex_flydsl_forward_template,
    maybe_append_flydsl_flex_attention_choice,
)
from torch._inductor.kernel.flex.flex_flydsl_mask import lower_flydsl_mask_graph
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.nn.attention.flex_attention import (
    and_masks,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)


def _has_gfx950_flydsl():
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950"
        and flydsl_utils.runtime_available()
    )


def _fake_choice_result(inputs):
    choices = []
    with (
        V.set_graph_handler(_fake_graph()),
        mock.patch(
            "torch._inductor.kernel.flex.flex_flydsl_config._is_gfx950_device",
            return_value=True,
        ),
        mock.patch.object(
            flex_flydsl_forward_template, "maybe_append_choice"
        ) as append,
    ):
        _, reason = maybe_append_flydsl_flex_attention_choice(
            choices, layout=mock.Mock(), **inputs
        )
    kwargs = append.call_args.kwargs if append.called else None
    return kwargs, reason


def _make_qkv(
    *,
    device="cuda",
    batch=1,
    query_heads=2,
    kv_heads=None,
    query_length=256,
    kv_length=None,
    qk_head_dim=128,
    value_head_dim=128,
    seed=0,
    strided=False,
):
    kv_heads = query_heads if kv_heads is None else kv_heads
    kv_length = query_length if kv_length is None else kv_length
    torch.manual_seed(seed)

    def make(heads, sequence, dimension):
        if not strided:
            return torch.randn(
                batch, heads, sequence, dimension, device=device, dtype=torch.bfloat16
            )
        return torch.randn(
            batch, sequence, heads, dimension, device=device, dtype=torch.bfloat16
        ).transpose(1, 2)

    return (
        make(query_heads, query_length, qk_head_dim),
        make(kv_heads, kv_length, qk_head_dim),
        make(kv_heads, kv_length, value_head_dim),
    )


class _FlyDSLFlexAttentionRuntimeMixin:
    def _require_runtime(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

    def _compare_forward(
        self,
        query,
        key,
        value,
        *,
        block_mask=None,
        reference_mask=None,
        scale=None,
        gqa=False,
        aux=False,
    ):
        def run(backend, selected_mask):
            def attention(query, key, value):
                kwargs = {
                    "block_mask": selected_mask,
                    "enable_gqa": gqa,
                    "kernel_options": {"BACKEND": backend},
                }
                if scale is not None:
                    kwargs["scale"] = scale
                if aux:
                    kwargs["return_aux"] = AuxRequest(lse=True, max_scores=True)
                return flex_attention(query, key, value, **kwargs)

            return run_and_get_code(
                torch.compile(attention, fullgraph=True), query, key, value
            )

        actual, code = run("FLYDSL", block_mask)
        expected, ref_code = run(
            "TRITON", reference_mask if reference_mask is not None else block_mask
        )
        torch.cuda.synchronize()
        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(ref_code))

        if not aux:
            # FlyDSL rounds scaled Q to bf16 before QK.
            self.assertEqual(actual, expected, atol=0.03, rtol=0.02)
            return actual

        output, aux = actual
        reference, reference_aux = expected
        # FlyDSL rounds scaled Q to bf16 before QK.
        self.assertEqual(output, reference, atol=0.03, rtol=0.02)
        self.assertEqual(aux.lse, reference_aux.lse, atol=0.03, rtol=0.01)
        self.assertEqual(aux.max_scores, reference_aux.max_scores, atol=0.03, rtol=0.01)
        return output, aux

    def _compare_created_mask(
        self,
        mask_mod,
        *,
        device,
        sequence_length=256,
        kv_length=None,
        mask_heads=1,
        scale=None,
        aux=False,
        **qkv_kwargs,
    ):
        query, key, value = _make_qkv(
            device=device,
            query_length=sequence_length,
            kv_length=kv_length,
            **qkv_kwargs,
        )
        block_mask = create_block_mask(
            mask_mod,
            query.size(0),
            mask_heads,
            sequence_length,
            key.size(2),
            device=device,
            BLOCK_SIZE=128,
        )
        return self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=scale,
            gqa=query.size(1) != key.size(1),
            aux=aux,
        )


class TestFlyDSLFlexAttentionBackend(_FlyDSLFlexAttentionRuntimeMixin, TestCase):
    @parametrize("qk_head_dim", [128, 192])
    @parametrize("partial", [False, True])
    def test_split_kv_reduction(self, device, qk_head_dim, partial):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            query_heads=4,
            query_length=1,
            kv_length=2048,
            qk_head_dim=qk_head_dim,
            seed=17,
        )
        counts = torch.arange(4, device=device, dtype=torch.int32).view(1, 4, 1)
        zero_counts = torch.zeros_like(counts)
        block_ids = torch.tensor([0, 5, 15], device=device, dtype=torch.int32)
        indices = block_ids.view(1, 1, 1, 3).expand(1, 4, 1, 3).contiguous()

        def mask_mod(batch_index, head_index, q_idx, kv_idx):
            return kv_idx % 2 == 0 if partial else kv_idx >= 0

        block_mask = BlockMask.from_kv_blocks(
            counts if partial else zero_counts,
            indices,
            zero_counts if partial else counts,
            indices,
            BLOCK_SIZE=128,
            mask_mod=mask_mod,
            seq_lengths=(1, 2048),
            compute_q_blocks=False,
        )

        output, aux = self._compare_forward(
            query, key, value, block_mask=block_mask, aux=True
        )
        positions = torch.arange(2048, device=device)
        active_blocks = (
            torch.arange(3, device=device)[None, :, None]
            < torch.arange(4, device=device)[:, None, None]
        )
        keep = (
            (
                active_blocks
                & ((positions // 128)[None, None, :] == block_ids[None, :, None])
            )
            .any(dim=1)
            .view(1, 4, 1, 2048)
        )
        if partial:
            keep = keep & (positions % 2 == 0)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), attn_mask=keep
        ).to(query.dtype)
        scores = (query.float() @ key.float().transpose(-1, -2)) * qk_head_dim**-0.5
        scores = scores.masked_fill(~keep, float("-inf"))
        # FlyDSL rounds scaled Q to bf16 before QK.
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)
        self.assertEqual(aux.lse, scores.logsumexp(-1), atol=0.025, rtol=0.025)
        self.assertEqual(aux.max_scores, scores.amax(-1), atol=0.025, rtol=0.025)
        self.assertEqual(output[:, 0], torch.zeros_like(output[:, 0]), atol=0, rtol=0)
        self.assertTrue(torch.isneginf(aux.lse[:, 0]).all())
        self.assertTrue(torch.isneginf(aux.max_scores[:, 0]).all())

    @parametrize("query_length", [1, 256])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_dense_default_uses_isolated_backend(self, device, query_length):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device, query_length=query_length, kv_length=256
        )
        compiled = torch.compile(
            lambda query, key, value: flex_attention(
                query, key, value, kernel_options={"BACKEND": "FLYDSL"}
            ),
            fullgraph=True,
        )
        with (
            mock.patch(
                "torch._inductor.kernel.flex.flex_attention._use_flex_decoding",
                side_effect=AssertionError("FlyDSL reached Triton decode selection"),
            ),
            mock.patch(
                "torch._inductor.kernel.flex.flex_attention."
                "flex_attention_template.maybe_append_choice",
                side_effect=AssertionError("FlyDSL registered a Triton candidate"),
            ),
        ):
            output = compiled(query, key, value)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float()
        ).to(query.dtype)
        # FlyDSL rounds the scaled Q to bf16 before QK.
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_computed_mask_capture(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device)
        ends = torch.arange(256, device=device, dtype=torch.int32)
        counts = torch.full((1, 1, 2), 2, device=device, dtype=torch.int32)
        indices = torch.tensor([[[[0, 1], [0, 1]]]], device=device, dtype=torch.int32)
        full_counts = torch.zeros_like(counts)
        full_indices = torch.zeros_like(indices)

        def attention(query, key, value, document_ends):
            computed_ends = document_ends + 1

            def mask_mod(batch_index, head_index, q_idx, kv_idx):
                return kv_idx < computed_ends[q_idx]

            block_mask = BlockMask.from_kv_blocks(
                counts,
                indices,
                full_counts,
                full_indices,
                BLOCK_SIZE=128,
                mask_mod=mask_mod,
                seq_lengths=(256, 256),
                compute_q_blocks=False,
            )
            return flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLYDSL"},
            )

        output = torch.compile(attention, fullgraph=True)(query, key, value, ends)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), is_causal=True
        ).to(query.dtype)
        # Match the bf16 scaled-Q precision used by the forward kernel.
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_backward_uses_triton_with_flydsl_forward(self, device):
        self._require_runtime()
        base_inputs = _make_qkv(device=device, query_length=128, kv_length=128, seed=23)
        torch.manual_seed(24)
        grad_output = torch.randn_like(base_inputs[2])

        def run(backend):
            inputs = tuple(
                tensor.detach().clone().requires_grad_() for tensor in base_inputs
            )
            compiled = torch.compile(
                lambda query, key, value: flex_attention(
                    query, key, value, kernel_options={"BACKEND": backend}
                ),
                fullgraph=True,
            )
            output, code = run_and_get_code(compiled, *inputs)
            output.backward(grad_output)
            return (
                output.detach(),
                tuple(tensor.grad.detach().clone() for tensor in inputs),
                "\n".join(code),
            )

        output, grads, code = run("FLYDSL")
        reference, ref_grads, ref_code = run("TRITON")
        self.assertIn("build_flex_attn_fwd_module", code)
        self.assertNotIn("build_flex_attn_fwd_module", ref_code)
        self.assertEqual(output, reference, atol=0.03, rtol=0.02)
        for grad, reference_grad in zip(grads, ref_grads):
            self.assertEqual(grad, reference_grad, atol=0.03, rtol=0.02)


instantiate_device_type_tests(
    TestFlyDSLFlexAttentionBackend, globals(), only_for=("cuda",)
)


@instantiate_parametrized_tests
class TestFlyDSLFlexAttentionConfig(TestCase):
    @parametrize("mask_buffer_count", [0, 2, 4])
    def test_precompile_uses_scheduler_metadata(self, mask_buffer_count):
        import jinja2

        from torch._inductor.codegen.flydsl.flydsl_scheduling import FlyDSLScheduling
        from torch._inductor.runtime.flydsl_cache import temporary_env

        input_names = []

        def define_kernel(*names):
            input_names.extend(names)
            return f"def kernel_main({','.join(names)}, output, stream):"

        source = jinja2.Template(flex_flydsl_forward_template.source).render(
            gen_defines=lambda: "",
            def_kernel=define_kernel,
            get_output=lambda: "output",
            kernel_name="kernel",
            MASK_BUFFER_COUNT=mask_buffer_count,
        )
        layout = SimpleNamespace(
            size=[1], stride=[1], dtype=torch.bfloat16, device=torch.device("cpu")
        )
        kernel = SimpleNamespace(
            _template_signature_defined=True,
            _template_input_args=[
                (f"arg_{name}", _FakeNode([1], [1])) for name in input_names
            ],
        )
        with mock.patch("torch.cuda.is_available", return_value=False):
            metadata = FlyDSLScheduling(scheduler=None)._build_precompile_metadata(
                kernel, SimpleNamespace(layout=layout)
            )
        self.assertIsNotNone(metadata)
        main = mock.Mock()
        namespace = dict(torch=torch, temporary_env=temporary_env, kernel_main=main)
        precompile_source = source[source.index("def kernel_precompile(") :]
        exec(compile(precompile_source, "<flex forward precompile>", "exec"), namespace)
        namespace["kernel_precompile"](**metadata)
        main.assert_called_once()
        self.assertEqual(len(main.call_args.args), len(input_names) + 1)
        self.assertEqual(main.call_args.kwargs, {"stream": 0})

    def test_unrealized_metadata_is_rejected(self):
        inputs = _supported_fake_forward_inputs()
        with (
            V.set_graph_handler(_fake_graph()),
            mock.patch.object(
                inputs["kv_indices"], "get_stride", side_effect=NotImplementedError
            ),
            mock.patch.object(
                flex_flydsl_forward_template, "maybe_append_choice"
            ) as append,
        ):
            appended, reason = maybe_append_flydsl_flex_attention_choice(
                [], layout=mock.Mock(), **inputs
            )
        self.assertFalse(appended)
        self.assertIn("requires statically known BlockMask metadata strides", reason)
        append.assert_not_called()


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
    def test_appends_supported_choices(
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
        kwargs, reason = _fake_choice_result(inputs)
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
        kwargs, reason = _fake_choice_result(inputs)
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
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNotNone(kwargs, reason)


class TestFlyDSLFlexAttention(_FlyDSLFlexAttentionRuntimeMixin, TestCase):
    @parametrize(
        "query_heads,kv_heads,sequence_length,captured",
        [
            (4, 2, 512, False),
            (64, 8, 2048, False),
            (8, 2, 4096, True),
            (16, 4, 8192, True),
        ],
    )
    def test_gfx950_forward_full_partial_gqa_and_empty_q_block(
        self, device, query_heads, kv_heads, sequence_length, captured
    ):
        self._require_runtime()

        head_dim = 192 if captured else 128
        query, key, value = _make_qkv(
            device=device,
            query_heads=query_heads,
            kv_heads=kv_heads,
            query_length=sequence_length,
            qk_head_dim=head_dim,
        )
        counts = torch.tensor(
            [[[1, 1, 1, 0] * (sequence_length // 512)]],
            device=device,
            dtype=torch.int32,
        )
        indices = torch.tensor(
            [[[[0], [1], [2], [0]] * (sequence_length // 512)]],
            device=device,
            dtype=torch.int32,
        )
        full_counts = torch.tensor(
            [[[0, 1, 1, 0] * (sequence_length // 512)]],
            device=device,
            dtype=torch.int32,
        )
        full_indices = torch.tensor(
            [[[[0]] * (sequence_length // 128)]], device=device, dtype=torch.int32
        )

        if captured:
            offsets = torch.zeros(sequence_length, device=device, dtype=torch.int32)

            def causal(batch_index, head_index, q_idx, kv_idx):
                del batch_index, head_index
                return q_idx >= kv_idx + offsets[q_idx]

        else:

            def causal(batch_index, head_index, q_idx, kv_idx):
                del batch_index, head_index
                return q_idx >= kv_idx

        block_mask = BlockMask.from_kv_blocks(
            counts,
            indices,
            full_counts,
            full_indices,
            BLOCK_SIZE=128,
            mask_mod=causal,
            seq_lengths=(sequence_length, sequence_length),
            compute_q_blocks=False,
        )
        output, aux = self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=head_dim**-0.5,
            gqa=True,
            aux=True,
        )
        for start in range(384, sequence_length, 512):
            self.assertEqual(output[:, :, start : start + 128].abs().max().item(), 0.0)
            self.assertTrue(torch.isneginf(aux.lse[:, :, start : start + 128]).all())
            self.assertTrue(
                torch.isneginf(aux.max_scores[:, :, start : start + 128]).all()
            )

    @parametrize(
        "sequence_length,document_starts", [(128, [0, 29, 83]), (8192, [0, 1701, 5387])]
    )
    def test_gfx950_forward_unequal_packed_documents(
        self, device, sequence_length, document_starts
    ):
        self._require_runtime()
        starts = torch.tensor(document_starts, device=device, dtype=torch.int32)
        document_ids = torch.bucketize(
            torch.arange(sequence_length, device=device), starts[1:], right=True
        ).to(torch.int32)

        def mask_mod(batch_index, head_index, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (kv_idx >= starts[document_ids[q_idx]])

        self._compare_created_mask(
            mask_mod,
            device=device,
            query_heads=16,
            kv_heads=4,
            sequence_length=sequence_length,
            qk_head_dim=192,
            strided=True,
            aux=True,
        )

    @parametrize("query_length", [1, 3, 4, 8, 16])
    def test_gfx950_public_api_per_kv_head_decode(self, device, query_length):
        self._require_runtime()
        batch, query_heads, kv_heads = 1, 64, 4
        kv_length, head_dim = 8192, 128
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            query_heads=query_heads,
            kv_heads=kv_heads,
            query_length=query_length,
            kv_length=kv_length,
            qk_head_dim=head_dim,
            seed=9,
        )
        counts = torch.ones(1, kv_heads, 1, device=device, dtype=torch.int32)
        indices = torch.zeros(1, kv_heads, 1, 16, device=device, dtype=torch.int32)
        indices[..., 0] = 63
        full_counts = torch.full((1, kv_heads, 1), 15, device=device, dtype=torch.int32)
        full_indices = torch.arange(15, device=device, dtype=torch.int32).view(
            1, 1, 1, 15
        ) + (
            torch.arange(kv_heads, device=device, dtype=torch.int32).view(
                1, kv_heads, 1, 1
            )
            * 8
        )

        q_offset = kv_length - query_length

        def bottom_right_causal(batch_index, head_index, q_idx, kv_idx):
            del batch_index, head_index
            return q_idx + q_offset >= kv_idx

        def make_mask(repeats):
            return BlockMask.from_kv_blocks(
                *(
                    tensor.repeat_interleave(repeats, dim=1)
                    for tensor in (counts, indices, full_counts, full_indices)
                ),
                BLOCK_SIZE=128,
                mask_mod=bottom_right_causal,
                seq_lengths=(query_length, kv_length),
                compute_q_blocks=False,
            )

        block_mask = make_mask(1)
        reference_mask = make_mask(query_heads // kv_heads)
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            reference_mask=reference_mask,
            scale=head_dim**-0.5,
            gqa=True,
            aux=True,
        )

    def test_gfx950_public_api_transposed_document_qk192_v128(self, device):
        self._require_runtime()

        batch, heads, sequence_length = 2, 2, 256
        qk_head_dim, value_head_dim = 192, 128
        document_end = torch.tensor(
            [[127] * 128 + [255] * 128, [63] * 64 + [191] * 128 + [255] * 64],
            device=device,
            dtype=torch.int32,
        )

        def document_causal(batch_index, head_index, q_idx, kv_idx):
            del head_index
            return (q_idx >= kv_idx) & (q_idx <= document_end[batch_index, kv_idx])

        output, _ = self._compare_created_mask(
            document_causal,
            device=device,
            batch=batch,
            query_heads=heads,
            sequence_length=sequence_length,
            qk_head_dim=qk_head_dim,
            value_head_dim=value_head_dim,
            seed=2,
            strided=True,
            scale=0.07,
            aux=True,
        )
        self.assertEqual(output.shape, (batch, heads, sequence_length, value_head_dim))
        self.assertEqual(output.stride()[-1], 1)

    @parametrize(
        "case", ["sliding_window", "wide_window", "window_decode", "head_dependent"]
    )
    def test_gfx950_public_api_supported_masks(self, device, case):
        self._require_runtime()
        if case == "head_dependent":

            def mask_mod(batch_index, head_index, query, kv_index):
                return (torch.add(query, 64, alpha=2) >= kv_index) & (
                    (torch.add(query, 64, alpha=2) - kv_index) % (2 + head_index) == 0
                )

            kwargs = {
                "batch": 2,
                "query_heads": 3,
                "sequence_length": 512,
                "seed": 8,
                "mask_heads": 3,
            }
        else:
            offset = 2044 if case == "window_decode" else 0
            width = 256 if case == "wide_window" else 96

            def mask_mod(batch_index, head_index, query, kv_index):
                return (query + offset >= kv_index) & (
                    query + offset - kv_index < width
                )

            sequence_length, kv_length = (
                (4, 2048) if offset else (512, 512) if width == 256 else (256, 256)
            )
            kwargs = {
                "query_heads": 4,
                "kv_heads": 2,
                "sequence_length": sequence_length,
                "kv_length": kv_length,
                "aux": True,
                "seed": 3 if case == "sliding_window" else 0,
            }
        self._compare_created_mask(mask_mod, device=device, **kwargs)

    @parametrize("sequence_length", [128, 512])
    def test_gfx950_public_api_composed_document_mask(self, device, sequence_length):
        self._require_runtime()
        document_ids = torch.arange(
            sequence_length, device=device, dtype=torch.int32
        ) // (sequence_length // 2)
        document_starts = torch.tensor(
            [0, sequence_length // 2], device=device, dtype=torch.int32
        )
        document_causal = and_masks(
            lambda batch_index, head_index, q_idx, kv_idx: q_idx >= kv_idx,
            lambda batch_index, head_index, q_idx, kv_idx: kv_idx
            >= document_starts[document_ids[q_idx]],
        )
        self._compare_created_mask(
            document_causal,
            device=device,
            sequence_length=sequence_length,
            seed=4,
            aux=True,
        )

    def test_gfx950_public_api_sliding_window_mask_lowering(self, device):
        self._require_runtime()

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_utils import (
            is_sliding_window_mask_program,
        )

        def check_match(mask_mod, expected):
            matched = []

            def check_window_lowering(*args, **kwargs):
                program, reason = lower_flydsl_mask_graph(*args, **kwargs)
                self.assertIsNotNone(program, reason)
                matched.append(
                    is_sliding_window_mask_program(
                        program.instructions, program.output, program.buffer_strides
                    )
                )
                return program, reason

            torch._dynamo.reset()
            with (
                torch._inductor.config.patch({"fx_graph_cache": False}),
                mock.patch(
                    "torch._inductor.kernel.flex.flex_flydsl_config.lower_flydsl_mask_graph",
                    side_effect=check_window_lowering,
                ) as lower,
            ):
                self._compare_created_mask(mask_mod, device=device, seed=3, aux=True)
                lower.assert_called_once()
            self.assertEqual(matched, [expected])

        # The two spellings lower to different programs, so both need covering:
        # and_masks() seeds a const_bool True that the bare lambda never emits.
        check_match(
            lambda batch_index, head_index, query, kv_index: (query >= kv_index)
            & (query - kv_index < 96),
            True,
        )
        check_match(
            and_masks(
                lambda batch_index, head_index, query, kv_index: query >= kv_index,
                lambda batch_index, head_index, query, kv_index: query - kv_index < 96,
            ),
            True,
        )
        # Unbounded causal must not match: it is the ragged-tail case the reversed
        # query-block dispatch exists for, and matching here would silently disable it.
        check_match(
            lambda batch_index, head_index, query, kv_index: query >= kv_index, False
        )

    def test_gfx950_auto_keeps_flydsl_opt_in(self, device):
        self._require_runtime()

        query, key, value = _make_qkv(device=device, seed=6)

        def causal(batch_index, head_index, q_idx, kv_idx):
            del batch_index, head_index
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal, 1, 1, 256, 256, device=device, BLOCK_SIZE=128
        )

        def run(query, key, value):
            return flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options={"BACKEND": "AUTO"},
            )

        torch._dynamo.reset()
        output, code = run_and_get_code(
            torch.compile(run, fullgraph=True), query, key, value
        )
        torch.cuda.synchronize()

        self.assertFalse(torch.isnan(output).any())
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(code))


instantiate_device_type_tests(TestFlyDSLFlexAttention, globals(), only_for=("cuda",))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
