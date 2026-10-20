"""Verify that inference hooks enhance only audio/video prompt tokens."""

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_classroom_m2kt_adapter import MemoryInjection


class DummyTextModel(nn.Module):
    def forward(self, inputs_embeds):
        return inputs_embeds


class DummyThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(audio_token_id=7, video_token_id=8,
                                      vision_start_token_id=9)
        self.model = DummyTextModel()

    def forward(self, input_ids, inputs_embeds):
        return self.model(inputs_embeds=inputs_embeds)


class AddOneToAV(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_index = None

    def forward(self, hidden, av_mask, audio_mask, video_mask, clip_index):
        self.clip_index = clip_index.detach().clone()
        assert torch.equal(av_mask, audio_mask | video_mask)
        return hidden + av_mask.unsqueeze(-1).to(hidden.dtype)


def test_only_av_tokens_are_changed_and_clip_order_is_preserved():
    if "with_kwargs" not in inspect.signature(nn.Module.register_forward_pre_hook).parameters:
        raise unittest.SkipTest("Inference hooks require the Omni virtual environment's PyTorch")
    thinker = DummyThinker()
    adapter = AddOneToAV()
    ids = torch.tensor([[1, 9, 7, 8, 2, 9, 8, 7, 3]])
    embeds = torch.zeros(1, ids.shape[1], 4)
    with MemoryInjection(thinker, adapter) as injection:
        result = thinker(input_ids=ids, inputs_embeds=embeds)
    expected = torch.zeros_like(embeds)
    expected[:, [2, 3, 6, 7]] = 1
    assert torch.equal(result, expected)
    assert injection.enhanced_calls == 1
    assert adapter.clip_index[0, [2, 3, 6, 7]].tolist() == [0, 0, 1, 1]


def test_no_av_tokens_does_not_invoke_adapter():
    if "with_kwargs" not in inspect.signature(nn.Module.register_forward_pre_hook).parameters:
        raise unittest.SkipTest("Inference hooks require the Omni virtual environment's PyTorch")
    thinker = DummyThinker()
    adapter = AddOneToAV()
    ids = torch.tensor([[1, 2, 3]])
    embeds = torch.zeros(1, 3, 4)
    with MemoryInjection(thinker, adapter) as injection:
        result = thinker(input_ids=ids, inputs_embeds=embeds)
    assert torch.equal(result, embeds)
    assert injection.enhanced_calls == 0


def test_generation_reuses_hook_without_enhancing_twice():
    if "with_kwargs" not in inspect.signature(nn.Module.register_forward_pre_hook).parameters:
        raise unittest.SkipTest("Inference hooks require the Omni virtual environment's PyTorch")
    thinker = DummyThinker()
    adapter = AddOneToAV()
    ids = torch.tensor([[9, 7, 8]])
    embeds = torch.zeros(1, ids.shape[1], 4)
    with MemoryInjection(thinker, adapter) as injection:
        first = thinker(input_ids=ids, inputs_embeds=embeds)
        second = thinker(input_ids=ids, inputs_embeds=embeds)
    assert injection.enhanced_calls == 1
    assert torch.all(first[:, 1:] == 1)
    assert torch.equal(second, embeds)
