"""Small M2KT-style memory adapter for classroom audio/video token features.

The original M2KT image extractor accepts radiographs, and its memory is tied to
medical labels.  This module instead accepts already extracted multimodal token
features.  Its memory slots are learned from classroom training data; they are
not a populated knowledge base or hand-labeled classroom concepts.

Only positions selected by ``av_mask`` receive a residual update.  In
particular, reference reports and their labels are never forward inputs.
"""

from typing import Optional

import torch
from torch import Tensor, nn


class ClassroomM2KTAdapter(nn.Module):
    """Read audio/video evidence into memory slots and write it back to AV tokens.

    Args:
        hidden_size: Width of the upstream model's token features.
        d_model: Width of the small trainable memory path.
        num_memory_slots: Number of learned memory vectors.
        num_heads: Attention heads in the memory path.
        max_clips: Number of distinct temporal clip IDs supported.
        dropout: Dropout probability in the adapter.

    Inputs to ``forward`` have shapes ``hidden_states: [B, S, hidden_size]``
    and ``av_mask: [B, S]``.  Optional modality masks and clip IDs also have
    shape ``[B, S]``.  A clip ID of -1 is allowed for non-AV positions.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        d_model: int = 256,
        num_memory_slots: int = 16,
        num_heads: int = 4,
        max_clips: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or d_model <= 0 or num_memory_slots <= 0 or max_clips <= 0:
            raise ValueError("feature widths, memory slots, and max_clips must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")

        self.hidden_size = hidden_size
        self.max_clips = max_clips
        self.input_proj = nn.Linear(hidden_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.audio_embedding = nn.Parameter(torch.zeros(d_model))
        self.video_embedding = nn.Parameter(torch.zeros(d_model))
        self.clip_embedding = nn.Embedding(max_clips, d_model)
        nn.init.zeros_(self.clip_embedding.weight)

        self.memory_slots = nn.Parameter(torch.empty(num_memory_slots, d_model))
        nn.init.normal_(self.memory_slots, mean=0.0, std=0.02)
        self.memory_read = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.memory_norm = nn.LayerNorm(d_model)
        self.token_read = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.token_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, hidden_size)
        self.output_gate = nn.Parameter(torch.tensor(-5.0))

    def forward(
        self,
        hidden_states: Tensor,
        av_mask: Tensor,
        audio_mask: Optional[Tensor] = None,
        video_mask: Optional[Tensor] = None,
        clip_index: Optional[Tensor] = None,
    ) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [B, S, hidden_size]")
        shape = hidden_states.shape[:2]
        if av_mask.shape != shape or av_mask.dtype != torch.bool:
            raise ValueError("av_mask must be a boolean [B, S] tensor")
        for name, mask in (("audio_mask", audio_mask), ("video_mask", video_mask)):
            if mask is not None:
                if mask.shape != shape or mask.dtype != torch.bool:
                    raise ValueError(f"{name} must be a boolean [B, S] tensor")
                if torch.any(mask & ~av_mask):
                    raise ValueError(f"{name} must be contained in av_mask")
        if clip_index is not None:
            if clip_index.shape != shape or clip_index.dtype not in (torch.int32, torch.int64):
                raise ValueError("clip_index must be an integer [B, S] tensor")
            selected_clips = clip_index[av_mask]
            if torch.any(selected_clips < 0) or torch.any(selected_clips >= self.max_clips):
                raise ValueError("AV clip IDs must lie in [0, max_clips)")

        valid_rows = av_mask.any(dim=1)
        if not torch.any(valid_rows):
            return hidden_states

        row_states = hidden_states[valid_rows].to(self.input_proj.weight.dtype)
        row_mask = av_mask[valid_rows]
        tokens = self.input_norm(self.input_proj(row_states))
        if audio_mask is not None:
            tokens = tokens + audio_mask[valid_rows].unsqueeze(-1).to(tokens.dtype) * self.audio_embedding
        if video_mask is not None:
            tokens = tokens + video_mask[valid_rows].unsqueeze(-1).to(tokens.dtype) * self.video_embedding
        if clip_index is not None:
            clip_ids = clip_index[valid_rows].clamp(min=0)
            tokens = tokens + self.clip_embedding(clip_ids) * row_mask.unsqueeze(-1)

        memory = self.memory_slots.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        evidence, _ = self.memory_read(
            memory, tokens, tokens, key_padding_mask=~row_mask, need_weights=False
        )
        memory = self.memory_norm(memory + evidence)
        context, _ = self.token_read(tokens, memory, memory, need_weights=False)
        delta = self.output_proj(self.token_norm(tokens + context))
        delta = delta * row_mask.unsqueeze(-1).to(delta.dtype)

        # A strongly closed gate preserves the upstream representation at
        # initialization while retaining a gradient path through every layer.
        updates = torch.zeros_like(hidden_states)
        updates[valid_rows] = (torch.sigmoid(self.output_gate) * delta).to(hidden_states.dtype)
        return hidden_states + updates
