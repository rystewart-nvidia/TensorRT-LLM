# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local/QA fused per-head RMSNorm and H3 partial split-half RoPE."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(
    X,
    W,
    COS,
    SIN,
    OUT,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    XB: tl.constexpr,
    XS: tl.constexpr,
    XH: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    seq = (rows // H) % S
    offsets = (rows // (S * H)) * XB + seq * XS + (rows % H) * XH
    x = tl.load(
        X + offsets[:, None] + cols[None, :],
        (rows[:, None] < N) & (cols[None, :] < D),
        0,
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    dtype = OUT.dtype.element_ty
    norm = x * inv[:, None]
    weight = tl.load(W + cols, cols < D, 0).to(tl.float32)
    norm = norm * weight[None, :]
    partner = tl.where(
        cols < R // 2, cols + R // 2, tl.where(cols < R, cols - R // 2, cols)
    )
    rotated = tl.gather(norm, tl.broadcast_to(partner[None, :], (ROWS, BLOCK)), 1)
    rotated = tl.where(cols[None, :] < R // 2, -rotated, rotated)
    cos = tl.load(
        COS + seq[:, None] * R + cols[None, :],
        (rows[:, None] < N) & (cols[None, :] < R),
        0,
    ).to(tl.float32)
    sin = tl.load(
        SIN + seq[:, None] * R + cols[None, :],
        (rows[:, None] < N) & (cols[None, :] < R),
        0,
    ).to(tl.float32)
    # The compiled H3 path materializes only the rotated sine term in BF16.
    first = norm * cos
    second = (rotated * sin).to(dtype).to(tl.float32)
    result = tl.where(cols[None, :] < R, first + second, norm)
    tl.store(
        OUT + rows[:, None] * D + cols[None, :],
        result,
        (rows[:, None] < N) & (cols[None, :] < D),
    )


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    b, s, h, d = x.shape
    result = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _norm_rope[(triton.cdiv(b * s * h, 8),)](
        x,
        weight,
        cos,
        sin,
        result,
        s,
        h,
        d,
        cos.shape[-1],
        b * s * h,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        eps,
        8,
        triton.next_power_of_2(d),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


@torch.library.custom_op("h3_qa::qk_norm_rope", mutates_args=())
def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qeps: float,
    keps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4 or q.shape[:2] != k.shape[:2]:
        raise ValueError("Expected Q/K with matching [B, S] dimensions")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype:
        raise ValueError("Prototype requires BF16 Q/K")
    if not q.is_cuda or any(t.device != q.device for t in (k, qw, kw, cos, sin)):
        raise ValueError("All tensors must share one CUDA device")
    if (
        qw.shape != (q.shape[-1],)
        or kw.shape != (k.shape[-1],)
        or not qw.is_contiguous()
        or not kw.is_contiguous()
    ):
        raise ValueError("Normalization weights must be contiguous per-head vectors")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("Head dimensions must be contiguous")
    if cos.ndim != 2 or cos.shape != sin.shape or cos.shape[0] != q.shape[1]:
        raise ValueError("RoPE tables must match the sequence")
    if not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("RoPE tables must be contiguous")
    rotary = cos.shape[-1]
    if rotary <= 0 or rotary % 2 or rotary > min(q.shape[-1], k.shape[-1]):
        raise ValueError("Invalid partial rotary dimension")
    return _launch(q, qw, cos, sin, qeps), _launch(k, kw, cos, sin, keps)


@fused_qk_norm_rope.register_fake
def _fake(q, k, qw, kw, cos, sin, qeps, keps):
    return torch.empty_like(q, memory_format=torch.contiguous_format), torch.empty_like(
        k, memory_format=torch.contiguous_format
    )


def install_fused_qk_rope() -> None:
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
        MiniMaxH3Attention,
    )

    original = MiniMaxH3Attention.forward

    def forward(
        self, hidden_states, rotary_emb=None, key_padding_mask=None, timestep=None
    ):
        if rotary_emb is None:
            return original(self, hidden_states, rotary_emb, key_padding_mask, timestep)
        b, s = hidden_states.shape[:2]
        q, k, v = self.get_qkv(hidden_states)
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
            return original(self, hidden_states, rotary_emb, key_padding_mask, timestep)
        q, k = fused_qk_norm_rope(
            q.view(b, s, self.local_num_attention_heads, self.head_dim),
            k.view(b, s, self.local_num_key_value_heads, self.head_dim),
            self.norm_q.weight,
            self.norm_k.weight,
            *rotary_emb,
            self.norm_q.variance_epsilon,
            self.norm_k.variance_epsilon,
        )
        result = self._attn_impl(
            q.flatten(2),
            k.flatten(2),
            v,
            key_padding_mask=key_padding_mask,
            timestep=timestep,
        )
        return self.to_out[0](result)

    MiniMaxH3Attention.forward = forward
